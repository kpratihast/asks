"""Normalize source teaching/exam slots. No course IDs, dates or answers are hard-coded."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import html
import re

DAYS = {'m':0,'mo':0,'mon':0,'monday':0,'t':1,'tu':1,'tue':1,'tues':1,'tuesday':1,
        'w':2,'we':2,'wed':2,'wednesday':2,'th':3,'thu':3,'thur':3,'thurs':3,'thursday':3,
        'f':4,'fr':4,'fri':4,'friday':4,'s':5,'sa':5,'sat':5,'saturday':5,'su':6,'sun':6,'sunday':6}
DAY_NAMES = ('MO','TU','WE','TH','FR','SA','SU')
MONTHS = {n:i+1 for i,n in enumerate(('january','february','march','april','may','june','july','august','september','october','november','december'))}


def weekdays(value):
    value = value.strip().lower().replace('&','/').replace('and','/')
    parts = re.findall(r'[a-z]+', value)
    if len(parts)==1 and parts[0] not in DAYS:
        parts = re.findall(r'th|su|m|t|w|f|s', parts[0])
        if ''.join(parts) != re.sub(r'[^a-z]', '', value):
            return None
    if not parts or any(p not in DAYS for p in parts):
        return None
    return tuple(sorted({DAYS[p] for p in parts}))


def day_family(days):
    # Normalize spelling, not membership: M/W is not silently treated as M/W/F.
    # Such an institute-specific equivalence needs an explicit source-backed rule.
    return days


def minutes(value, *, clock='24h'):
    value=value.strip().lower()
    m=re.fullmatch(r'(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?',value)
    if not m:return None
    h,minute=int(m[1]),int(m[2] or 0)
    if minute>=60:return None
    if m[3]:
        if not 1<=h<=12:return None
        h=h%12+(12 if m[3]=='pm' else 0)
    elif clock=='academic12' and 1<=h<=7:
        h+=12
    if not 0<=h<=23:return None
    return h*60+minute


def time_range(value, *, clock='24h', shorthand=False):
    parts=re.split(r'\s*(?:-|–|—|\bto\b)\s*',value.strip(),flags=re.I)
    if len(parts)==1 and shorthand:
        start=minutes(parts[0],clock=clock)
        return (start,start+60) if start is not None and start<23*60 else None
    if len(parts)!=2:return None
    start,end=(minutes(x,clock=clock) for x in parts)
    return (start,end) if start is not None and end is not None and start<end else None


def hm(value):return f'{value//60:02d}:{value%60:02d}'


@dataclass(frozen=True)
class TeachingSlot:
    days: tuple[int,...]
    start: int
    end: int

    def key(self, term):
        return f'TeachingSlot:{term}:{",".join(DAY_NAMES[d] for d in day_family(self.days))}:{hm(self.start)}-{hm(self.end)}'


def slot(days,timing, *, clock='24h', shorthand=False):
    ds,ts=weekdays(days),time_range(timing,clock=clock,shorthand=shorthand)
    return TeachingSlot(ds,*ts) if ds and ts else None


def pattern_slot(text):
    m=re.fullmatch(r'\s*([A-Za-z/ ,&]+?)\s+(\d.*)',text)
    return slot(m[1],m[2],clock='academic12',shorthand=True) if m else None


def fields(text):
    return {k.strip().lower():v.strip() for part in re.split(r';\s|\s\|\s',text)
            if ': ' in part for k,v in [part.split(': ',1)]}


def teaching_slot(text):
    f=fields(text)
    days=next((v for k,v in f.items() if k in ('days','days of week','days of the week','day')),None)
    times=next((v for k,v in f.items() if k in ('timings','timing','time','class time')),None)
    return slot(days,times) if days and times else None


ROW = re.compile(r'^\s*(\d+)\.\s+(.+?)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+'
                 r'([A-Za-z]+)\s+(\d{1,2})\s+(Forenoon|Afternoon)\s*$',re.I)
SESSION = re.compile(r'(Forenoon|Afternoon)\s*:\s*(\d{1,2}[:.]\d{2}\s*(?:AM|PM))\s*[–—-]\s*'
                     r'(\d{1,2}[:.]\d{2}\s*(?:AM|PM))',re.I)


def exam_table_html(layout_pages, title, term, academic_term):
    """Convert recognized numbered exam lookup rows to individually citable sections.

    The ISO date, hours and alternatives come only from PDF text. Unrecognized
    layouts return None; callers retain text instead of inventing a timetable.
    """
    year=int(academic_term.split('-')[0]);text='\n'.join(layout_pages)
    sessions={m[1].lower():time_range(m[2]+'-'+m[3]) for m in SESSION.finditer(text)}
    rows=[];remainder=[]
    for line in text.splitlines():
        m=ROW.match(line)
        if not m:
            if line.strip():remainder.append(line.strip())
            continue
        alternatives=[p.strip() for p in m[2].split(';')]
        session=m[6].lower();times=sessions.get(session)
        if not times or any(pattern_slot(p) is None for p in alternatives):return None
        month=MONTHS.get(m[4].lower())
        if month is None:return None
        exam_date=date(year,month,int(m[5]))
        if exam_date.strftime('%A').lower()!=m[3].lower():raise ValueError('PDF exam date and weekday disagree')
        rows.append((m[1],alternatives,exam_date,session,times))
    if not rows:return None
    if len({n for n,*_ in rows})!=len(rows):raise ValueError('Duplicate exam row numbers need review')
    parts=['<html><head><title>'+html.escape(title)+'</title></head><body><main><h1>'+html.escape(title)+'</h1><h2>'+html.escape(term)+'</h2>']
    for number,patterns,exam_date,session,times in rows:
        parts.append(f'<h3>Exam lookup row {number}</h3><p>'+html.escape(
            f'Course time slots: {" / ".join(patterns)}; Final examination date: {exam_date.isoformat()} ({exam_date.strftime("%A")}); '
            f'Examination session: {session.capitalize()}; Examination time: {hm(times[0])}-{hm(times[1])}')+'</p>')
    parts.append('<h3>Timetable instructions and session hours</h3>')
    parts.extend('<p>'+html.escape(line)+'</p>' for line in remainder)
    return ''.join(parts)+'</main></body></html>'


def exam_lookup(text, term):
    f=fields(text)
    patterns=f.get('course time slots','')
    when=re.match(r'(\d{4}-\d{2}-\d{2})\b',f.get('final examination date',''))
    timing=time_range(f.get('examination time',''))
    if not patterns or not when or not timing or not term:return None
    try:date.fromisoformat(when[1])
    except ValueError:return None
    slots=[pattern_slot(p.strip()) for p in patterns.split(' / ')]
    if not all(slots):return None
    destination=f'ExamSession:{term}:{when[1]}:{hm(timing[0])}-{hm(timing[1])}'
    return slots,destination
