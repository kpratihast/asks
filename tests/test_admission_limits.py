"""Global/tenant queue bounds, cancellation, HTTP bursts and cloud limit configuration."""
import asyncio,json,sys
from pathlib import Path
import httpx
import pytest
from askiisc_serving.admission import Admission,Rejected
from askiisc_serving.app import create_app
from askiisc_serving.generator import Generator
from askiisc_serving.settings import Settings
from test_serving import FakeRetrieval,sse_body
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deploy/aws'))
import release_lib,runtime


def test_global_saturation_queues_instead_of_bypassing_the_tenant_queue():
    async def check():
        adm=Admission(100,1,1,.5,1);first=await adm.acquire('key','public')
        waiter=asyncio.create_task(adm.acquire('key','public'));await asyncio.sleep(.01)
        assert adm.snapshot()['tenants']['public']=={'active':1,'waiting':1}
        with pytest.raises(Rejected) as full:await adm.acquire('key','public')
        assert full.value.reason=='tenant_queue_full'
        await first.release();second=await waiter;await second.release()
        assert adm.snapshot()['global_active']==0 and adm.snapshot()['rejected_global']==0
    asyncio.run(check())


def test_waiters_across_tenants_never_exceed_the_shared_global_limit():
    async def check():
        adm=Admission(100,1,5,2,2);peak=0
        async def job(i):
            nonlocal peak
            slot=await adm.acquire(str(i),str(i%8))
            peak=max(peak,adm.global_active)
            try:await asyncio.sleep(.002)
            finally:await slot.release()
        await asyncio.gather(*(job(i) for i in range(32)))
        assert peak==2 and adm.global_active==0
        assert adm.snapshot()['admitted']==32
        assert all(t['waiting']==t['active']==0 for t in adm.snapshot()['tenants'].values())
    asyncio.run(check())


def test_cancellation_and_timeout_remove_queue_tickets_and_release_is_idempotent():
    async def check():
        adm=Admission(100,1,1,.02,1);first=await adm.acquire('key','public')
        cancelled=asyncio.create_task(adm.acquire('key','public'));await asyncio.sleep(.005);cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):await cancelled
        assert adm.snapshot()['tenants']['public']['waiting']==0
        with pytest.raises(Rejected) as timeout:await adm.acquire('key','other')
        assert timeout.value.reason=='queue_timeout'
        assert adm.snapshot()['tenants']['other']['waiting']==0
        waiting=asyncio.create_task(adm.acquire('key','public'));await asyncio.sleep(.005)
        await asyncio.gather(first.release(),first.release())
        second=await waiting;assert adm.global_active==1
        await second.release();assert adm.global_active==0
    asyncio.run(check())


def test_waiters_do_not_get_overtaken_by_a_new_request():
    async def check():
        adm=Admission(100,1,3,1,1);held=await adm.acquire('key','public');order=[]
        async def job(n):
            slot=await adm.acquire('key','public');order.append(n)
            await asyncio.sleep(.001);await slot.release()
        tasks=[asyncio.create_task(job(1)),asyncio.create_task(job(2))];await asyncio.sleep(.005)
        await held.release();tasks.append(asyncio.create_task(job(3)))
        await asyncio.gather(*tasks);assert order==[1,2,3]
    asyncio.run(check())


def test_http_c20_q10_burst_admits_twenty_queues_ten_and_rejects_one():
    async def check():
        release=asyncio.Event();started=asyncio.Event();calls=0
        async def model(request):
            nonlocal calls
            if request.url.path=='/v1/models':return httpx.Response(200,json={'data':[{'id':'fake'}]})
            calls+=1
            if calls==20:started.set()
            await release.wait()
            return httpx.Response(200,content=sse_body('Yogesh Simmhan [S1].'),headers={'content-type':'text/event-stream'})
        settings=Settings(public_chat=True,email_auth_enabled=False,vllm_url='http://fake',served_model='fake',
                          rate_per_min=100,tenant_concurrency=20,tenant_queue=10,global_concurrency=20,queue_wait_s=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as upstream:
            app=create_app(settings,retrieval=FakeRetrieval(),generator=Generator(settings,client=upstream))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                tasks=[asyncio.create_task(client.post('/v1/ask',json={'question':'Who teaches DS 252?'})) for _ in range(31)]
                try:
                    await asyncio.wait_for(started.wait(),2)
                    for _ in range(100):
                        state=app.state.admission.snapshot()
                        if state['tenants']['public']['waiting']==10 and state['rejected_queue_full']==1:break
                        await asyncio.sleep(.001)
                    assert state['global_active']==20 and state['tenants']['public']['waiting']==10
                    assert state['rejected_queue_full']==1 and state['rejected_rate']==0
                finally:release.set()
                results=await asyncio.wait_for(asyncio.gather(*tasks),5)
                assert sum(r.status_code==200 for r in results)==30
                assert sum(r.status_code==429 for r in results)==1
                denied=next(r for r in results if r.status_code==429)
                assert denied.json()['reason']=='tenant_queue_full' and int(denied.headers['retry-after'])>=1
                assert all(r.json()['status']=='answered' for r in results if r.status_code==200)
                state=(await client.get('/readyz')).json()['admission']
                assert state['global_active']==0 and state['tenants']['public']['waiting']==0
                assert state['limits']['tenant_concurrency']==20 and state['limits']['tenant_queue']==10
                assert (await client.get('/readyz')).json()['anonymous_rate_scope']=='shared'
    asyncio.run(check())


def cloud(**values):
    return release_lib.runtime_config({'VLLM_URL':'https://models.example.org/v1','SERVED_MODEL':'test',
                                       'EMAIL_AUTH_ENABLED':False,**values})


def test_cloud_limits_are_configurable_but_defaults_remain_conservative(tmp_path):
    default=cloud();assert 'GLOBAL_CONCURRENCY' not in default  # Preserve existing SSM digests/model-switch compatibility.
    release=tmp_path/'release';release.mkdir();(release/'release.json').write_text('{"release_id":"r"}')
    env=runtime.environment(release,default,tmp_path/'state','chat.example.org')
    assert {k:env[k] for k in ['GLOBAL_CONCURRENCY','TENANT_CONCURRENCY','TENANT_QUEUE','QUEUE_WAIT_S']}=={
        'GLOBAL_CONCURRENCY':'1','TENANT_CONCURRENCY':'1','TENANT_QUEUE':'4','QUEUE_WAIT_S':'120'}
    configured=cloud(RATE_PER_MIN=60,GLOBAL_CONCURRENCY=20,TENANT_CONCURRENCY=20,TENANT_QUEUE=10,QUEUE_WAIT_S=30)
    env=runtime.environment(release,configured,tmp_path/'state','chat.example.org')
    assert {k:env[k] for k in ['GLOBAL_CONCURRENCY','TENANT_CONCURRENCY','TENANT_QUEUE','QUEUE_WAIT_S','RATE_PER_MIN']}=={
        'GLOBAL_CONCURRENCY':'20','TENANT_CONCURRENCY':'20','TENANT_QUEUE':'10','QUEUE_WAIT_S':'30','RATE_PER_MIN':'60'}


@pytest.mark.parametrize('values',[
    {'GLOBAL_CONCURRENCY':0},{'GLOBAL_CONCURRENCY':33},{'TENANT_CONCURRENCY':-1},{'TENANT_CONCURRENCY':True},
    {'TENANT_QUEUE':-1},{'TENANT_QUEUE':101},{'RATE_PER_MIN':101},{'TENANT_CONCURRENCY':1.5},
    {'QUEUE_WAIT_S':0},{'QUEUE_WAIT_S':float('nan')},{'QUEUE_WAIT_S':float('inf')},{'QUEUE_WAIT_S':301},
])
def test_invalid_cloud_admission_limits_are_rejected(values):
    with pytest.raises(ValueError):cloud(**values)


def test_disabling_the_queue_preserves_an_explicit_busy_response():
    async def check():
        adm=Admission(100,1,0,1,1);slot=await adm.acquire('key','public')
        with pytest.raises(Rejected) as busy:await adm.acquire('key','other')
        assert busy.value.reason=='server_busy'
        await slot.release()
    asyncio.run(check())
