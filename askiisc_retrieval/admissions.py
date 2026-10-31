"""Source-format normalization for a bounded institute PG + CDS admissions snapshot.

Original pages are cached with checksums. This layer preserves programme/category
associations, removes unrelated department panels and never invents admissions facts.
"""
from __future__ import annotations
import html,re
from urllib.parse import urljoin
from bs4 import BeautifulSoup,Tag

CDS = re.compile(r'computational\s*(?:&|and)\s*data\s*sciences?',re.I)
FOOTER=re.compile(r'^(?:recent news|contact us|shortcuts|general information|arting science|campus life|did you know\??|artsy research)$',re.I)

def text(node):return re.sub(r'\s+',' ',node.get_text(' ',strip=True)).strip()

def table_grid(table):
    """Expand source rowspan/colspan cells before associating values with headers."""
    grid=[];carry={}
    for tr in table.find_all('tr'):
        row={i:v for i,(v,n) in carry.items()};next_carry={i:(v,n-1) for i,(v,n) in carry.items() if n>1};col=0
        for cell in tr.find_all(['td','th'],recursive=False):
            while col in row:col+=1
            cs,rs=int(cell.get('colspan',1)),int(cell.get('rowspan',1))
            if not 1<=cs<=30 or not 1<=rs<=100:raise ValueError('Unrecognized table span')
            value=text(cell)
            for x in range(col,col+cs):
                if x in row:raise ValueError('Overlapping table cells')
                row[x]=value
                if rs>1:next_carry[x]=(value,rs-1)
            col+=cs
        grid.append([row.get(x,'') for x in range(max(row,default=-1)+1)]);carry=next_carry
    return grid

def fees_table(table):
    rows=table_grid(table)
    if not rows:return ''
    caption=rows[0][0]
    if not re.search(r'Fee structure for (?:Regular Research|Course Programme)',caption,re.I):return ''
    if len(rows)<4 or 'Category' not in ' '.join(rows[1]):raise ValueError('Fee column labels require review')
    categories,programmes=rows[1:3];parts=[]
    for column in range(1,min(len(categories),len(programmes))):
        if not re.search(r'Ph\.?D|M\.?\s*Tech',programmes[column],re.I) or 'Int.' in programmes[column]:continue
        heading=f'{caption} — {programmes[column]} — {categories[column]}'
        parts.append('<h3>'+html.escape(heading)+'</h3>')
        for row in rows[3:]:
            if len(set(row))==1:
                if row and row[0]:parts.append('<p>'+html.escape(row[0])+'</p>')
                continue
            if column>=len(row):raise ValueError('Incomplete fee row')
            parts.append('<p>'+html.escape(row[0]+': '+row[column])+'</p>')
    return ''.join(parts)

def normalize_html(body,url,mode):
    soup=BeautifulSoup(body,'lxml')
    main=soup.select_one('.col-md-9') if mode!='cds' else soup.select_one('.entry-content') or soup.select_one('main') or soup.body
    if main is None:raise ValueError('Admissions content container needs review')
    main=BeautifulSoup(str(main),'lxml')
    # Only department panels are scoped; common eligibility/selection panels remain complete.
    found_cds=False
    if mode=='programmes':
        for listing in list(main.select('ul.resp-tabs-list')):
            panels=listing.find_next_sibling('div',class_='resp-tabs-container')
            if panels is None:raise ValueError('Admissions tabs need review')
            tabs=panels.find_all('div',class_='fruitful_tab',recursive=False)
            labels=listing.find_all('li',recursive=False)
            department_panels=any('Discipline:' in text(tab) or 'Department/Centre/Unit:' in text(tab) for tab in tabs)
            if department_panels:
                # The live M.Tech markup has an unwrapped Instrumentation panel, so
                # positional label/tab pairing would associate the wrong programme.
                kept=[tab for tab in tabs if CDS.search(text(tab)[:300])]
                if any(CDS.search(text(label)) for label in labels) and not kept:
                    raise ValueError('CDS programme panel missing; source format needs review')
                found_cds = found_cds or bool(kept)
                for node in list(panels.children):
                    if isinstance(node,Tag) and node not in kept:node.decompose()
                for tab in kept:
                    heading=main.new_tag('h2');heading.string='Computational and Data Sciences (CDS)';tab.insert(0,heading)
            else:
                if len(tabs)!=len(labels):raise ValueError('Admissions tab/label mismatch')
                for label,tab in zip(labels,tabs):
                    heading=main.new_tag('h2');heading.string=text(label);tab.insert(0,heading)
            listing.decompose()
    if mode=='programmes':
        if not found_cds:raise ValueError('No CDS programme panel found')
        # Empty headings above department tabs should not label the common rules.
        for heading in list(main.find_all(['h1','h2'])):
            if re.fullmatch(r'Ph\.?D\.? Science\s*\*?', text(heading), re.I):heading.decompose()
    if mode=='cfti':
        if not any(CDS.search(text(h)) for h in main.find_all('h2')):raise ValueError('CFTI CDS panel missing')
        active=True
        for node in list((main.body or main).find('div',class_='col-md-9').children):
            if isinstance(node,Tag) and node.name=='h2':active=bool(CDS.search(text(node)))
            if not active:node.extract()
    if mode=='fees':
        for i,table in enumerate(list(main.find_all('table'))):
            if i==0:
                # Preserve the programme labels next to monthly fellowship values.
                rows=table_grid(table);frag='<h2>Monthly scholarship and fellowship amounts</h2>'
                for row in rows[1:]:
                    if len(row)==2 and re.search(r'M\.?\s*Tech|Ph\.?D',row[0],re.I) and 'Integrated' not in row[0]:
                        frag+='<h3>'+html.escape(row[0])+'</h3><p>Programme: '+html.escape(row[0])+'; Scholarship/fellowship per month: '+html.escape(row[1])+'</p>'
            else:frag=fees_table(table)
            table.replace_with(BeautifulSoup(frag,'html.parser'))
        # UG application fees have their own scope and are outside this snapshot.
        for p in list(main.find_all('p')):
            if re.search(r'^For UG',text(p),re.I):
                nxt=p.find_next_sibling()
                if nxt and nxt.name=='ul':nxt.decompose()
                p.decompose()
    if mode=='contacts':
        for table in main.find_all('table'):
            rows=table.find_all('tr');kept=[]
            for row in rows:
                if CDS.search(text(row)):kept.append(str(row))
            table.replace_with(BeautifulSoup('<h2>Computational and Data Sciences contact</h2>'+''.join('<p>'+html.escape(text(BeautifulSoup(row,'html.parser')) )+'</p>' for row in kept),'html.parser'))
    if mode=='faq':
        in_ug=False
        for p in list(main.find_all('p')):
            strong=p.find('strong',recursive=False)
            if strong and text(p)==text(strong):
                title=text(p)
                if re.search(r'Undergraduate',title,re.I):in_ug=True
                elif re.search(r'Postgraduate|Research Program|General Information',title,re.I):in_ug=False
                heading=main.new_tag('h3' if title.endswith('?') else 'h2');heading.string=title;p.replace_with(heading)
                if in_ug:heading.decompose()
            elif in_ug:p.decompose()
    if mode=='cds':
        for heading in list(main.find_all(['h1','h2','h3','h4'])):
            if FOOTER.match(text(heading)):
                for nxt in list(heading.next_siblings):
                    if isinstance(nxt,Tag) and re.fullmatch(r'h[1-4]',nxt.name):break
                    if isinstance(nxt,Tag):nxt.decompose()
                heading.decompose()
    # Side menus and campus widgets are outside .col-md-9, and never become evidence.
    for node in main.select('script,style,nav,header,footer,form,iframe,.resp-tabs-list,.randomBox1'):node.decompose()
    for a in main.find_all('a',href=True):
        href=urljoin(url,a['href'])
        if href.startswith(('https://','http://')) and text(a).lower() in {'here','click here','click here to view','online admissions portal','iisc admissions website','iisc online admissions application'}:
            a.append(' ('+href+')')
    title=next((text(h) for h in main.find_all('h1') if text(h)!='Admissions'),url)
    return '<html><head><title>'+html.escape(title)+'</title></head><body><main>'+str(main.body or main)+'</main></body></html>'

def pdf_html(path,title,*,max_pages=6):
    from pypdf import PdfReader
    reader=PdfReader(path)
    if len(reader.pages)>max_pages:raise ValueError('PDF page boundary exceeded')
    parts=['<h1>'+html.escape(title)+'</h1>']
    for n,page in enumerate(reader.pages,1):
        value=page.extract_text(extraction_mode='layout') or ''
        if len(value.strip())<60:raise ValueError('Scanned/empty PDF needs reviewed OCR')
        parts+=['<h2>'+html.escape(title)+f' — PDF page {n}</h2>']
        # Keep layout columns within each source line. Page and notice title remain citable.
        parts+=['<p>'+html.escape(line.strip())+'</p>' for line in value.splitlines() if line.strip()]
    return '<html><head><title>'+html.escape(title)+'</title></head><body><main>'+''.join(parts)+'</main></body></html>'


def cached_pages(profile,cache):
    import hashlib,json
    from pathlib import Path
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    from askiisc_retrieval.ingest_stub import PageInput,MIN_CONTENT_CHARS
    from askiisc_retrieval.parse import parse_html
    from askiisc_retrieval.chunking import chunk_page
    limits=profile['limits'];specs=profile['sources']
    if len({spec['url'] for spec in specs})!=len(specs):raise ValueError('Duplicate source URL')
    if len(specs)>limits['sources']:raise ValueError('Source boundary exceeded')
    records={d['url']:d for d in json.loads((cache/'download.json').read_text())}
    pages=[];report=[];total=0;pdf_pages=0;estimated=0
    (cache/'normalized').mkdir(exist_ok=True)
    for spec in specs:
        rec=records[spec['url']];name=rec['file']
        if Path(name).name!=name:raise ValueError('Invalid raw cache filename')
        path=cache/'raw'/name
        if digest(path)!=rec['sha256']:raise ValueError('Source checksum changed: '+spec['url'])
        total+=path.stat().st_size
        if total>limits['total_bytes']:raise ValueError('Byte boundary exceeded')
        if spec['mode']=='skip':
            report.append({**spec,'status':'skipped','reason':spec['reason']});continue
        if spec['mode']=='pdf':
            from pypdf import PdfReader
            pdf_pages+=len(PdfReader(path).pages)
            if pdf_pages>limits['pdf_pages']:raise ValueError('PDF page boundary exceeded')
            body=pdf_html(path,spec['label'],max_pages=limits['pdf_pages'])
        else:body=normalize_html(path.read_bytes(),spec['url'],spec['mode'])
        parsed=parse_html(spec['url'],body)
        if sum(len(s.text) for s in parsed.sections)<MIN_CONTENT_CHARS:raise ValueError('Empty source needs review: '+spec['url'])
        chunks=chunk_page(parsed,version_id='preview',document_id='preview',scope='public',doc_type='admissions',department=spec['department'],embedding_model='preview')
        estimated+=len(chunks)
        if estimated>limits['new_chunks']:raise ValueError('Chunk boundary exceeded: split at programme/source boundary')
        (cache/'normalized'/(name+'.html')).write_text(body)
        pages.append(PageInput(spec['url'],body,'admissions','public',spec['department']))
        report.append({**spec,'status':'included','sha256':rec['sha256'],'sections':len(parsed.sections),'estimated_chunks':len(chunks),'words':sum(len(s.text.split()) for s in parsed.sections)})
    return pages,{'sources':report,'raw_bytes':total,'pdf_pages':pdf_pages,'estimated_new_chunks':estimated,'boundary':profile['boundary']}
