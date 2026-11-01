#!/usr/bin/env python3
"""Offline admission-only burst. Synthetic held slots: no HTTP, retrieval or model calls.

Example: python bench/admission_probe.py --concurrency 20 --queue 10 --requests 31
This checks admission policy, not LLM/EC2 throughput. Request attempts consume rate tokens.
"""
from __future__ import annotations
import argparse,asyncio,json,sys,time
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from askiisc_serving.admission import Admission,Rejected


async def probe(concurrency=20,queue=10,requests=31,rate_per_min=100,hold_ms=100,admission_class=Admission):
    adm=admission_class(rate_per_min,concurrency,queue,10,concurrency)
    start=asyncio.Event();outcomes=Counter();peak_active=peak_waiting=0
    def sample():
        nonlocal peak_active,peak_waiting
        state=adm.snapshot();peak_active=max(peak_active,state['global_active'])
        peak_waiting=max(peak_waiting,sum(t['waiting'] for t in state['tenants'].values()))
    async def job():
        await start.wait()
        try:
            slot=await adm.acquire('shared-public-key','public')
        except Exception as exc:
            if hasattr(exc,'reason'):outcomes[exc.reason]+=1;return
            raise
        sample()
        try:await asyncio.sleep(hold_ms/1000)
        finally:await slot.release()
        outcomes['completed']+=1
    tasks=[asyncio.create_task(job()) for _ in range(requests)]
    began=time.monotonic();start.set()
    while any(not task.done() for task in tasks):
        sample();await asyncio.sleep(.001)
    await asyncio.gather(*tasks);sample()
    return {'scope':'offline admission only; synthetic work, no HTTP/retrieval/LLM',
            'concurrency':concurrency,'queue':queue,'rate_per_min':rate_per_min,'requests':requests,
            'peak_active':peak_active,'peak_waiting':peak_waiting,'outcomes':dict(outcomes),
            'wall_ms':round((time.monotonic()-began)*1000,1),'final':adm.snapshot()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--concurrency',type=int,default=20);p.add_argument('--queue',type=int,default=10);p.add_argument('--requests',type=int,default=31);p.add_argument('--rate-per-min',type=int,default=100);p.add_argument('--hold-ms',type=int,default=100);a=p.parse_args()
    if not 1<=a.requests<=1000 or a.hold_ms<1:p.error('Use 1–1000 requests and positive --hold-ms')
    print(json.dumps(asyncio.run(probe(a.concurrency,a.queue,a.requests,a.rate_per_min,a.hold_ms)),indent=2))
if __name__=='__main__':main()
