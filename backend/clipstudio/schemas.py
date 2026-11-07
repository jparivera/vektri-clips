from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Keyframe(Strict):
    time: float = Field(ge=0, le=86400)
    x: float = Field(default=.5, ge=0, le=1)
    y: float = Field(default=.5, ge=0, le=1)

class Crop(Strict):
    mode: Literal['fit', 'crop', 'split'] = 'fit'
    x: float = Field(default=.5, ge=0, le=1)
    y: float = Field(default=.5, ge=0, le=1)
    zoom: float = Field(default=1, ge=1, le=3)
    second_x: float = Field(default=.75, ge=0, le=1)
    keyframes: list[Keyframe] = Field(default_factory=list, max_length=2000)

    @model_validator(mode='after')
    def ordered(self):
        if any(a.time >= b.time for a,b in zip(self.keyframes, self.keyframes[1:])):
            raise ValueError('Crop keyframes must be ordered with distinct times')
        return self

class Segment(Strict):
    id: str = Field(min_length=1,max_length=80,pattern=r'^[a-zA-Z0-9_-]+$')
    start: float = Field(ge=0,le=86400)
    end: float = Field(gt=0,le=86400)
    crop: Crop = Field(default_factory=Crop)
    gain_db: float = Field(default=0, ge=-40, le=12)

    @model_validator(mode='after')
    def bounds(self):
        # Source timestamps subtract with normal binary floating-point error.
        if self.end - self.start < .04 - 1e-9:
            raise ValueError('Each segment must contain at least 40 milliseconds')
        return self

class Output(Strict):
    width: int = Field(default=720,ge=144,le=1920,multiple_of=2)
    height: int = Field(default=1280,ge=144,le=1920,multiple_of=2)
    fps: int = Field(default=30,ge=12,le=60)
    crf: int = Field(default=22,ge=16,le=32)

class Word(Strict):
    word: str = Field(min_length=1,max_length=500)
    start: float = Field(ge=0,le=86400)
    end: float = Field(gt=0,le=86400)

    @model_validator(mode='after')
    def bounds(self):
        if self.end <= self.start or '\x00' in self.word:
            raise ValueError('Caption words need valid timing and text')
        return self

class Captions(Strict):
    enabled: bool = True
    burn_in: bool = False
    style: Literal['clean','boxed','accent'] = 'clean'
    font_size: int = Field(default=42,ge=12,le=100)
    margin: int = Field(default=100,ge=10,le=400)
    alignment: Literal[1,2,3,7,8,9] = 2
    background: bool = False
    highlight: bool = False
    font_family: Literal['DejaVu Sans','DejaVu Serif','DejaVu Sans Mono'] = 'DejaVu Sans'
    color: str = Field(default='#FFFFFF',pattern=r'^#[0-9a-fA-F]{6}$')
    highlight_color: str = Field(default='#D3EC78',pattern=r'^#[0-9a-fA-F]{6}$')
    stroke_width: float = Field(default=2,ge=0,le=8)
    words: list[Word] = Field(default_factory=list,max_length=100000)

    @model_validator(mode='after')
    def ordered(self):
        if any(a.end > b.start + .001 for a,b in zip(self.words, self.words[1:])):
            raise ValueError('Caption word timings must be ordered and non-overlapping')
        return self

class Audio(Strict):
    normalize: bool = False
    fade: float = Field(default=0,ge=0,le=1)
    stream_index: int = Field(default=0,ge=0,le=31)

class Timeline(Strict):
    schema_version: Literal[1] = 1
    segments: list[Segment] = Field(min_length=1,max_length=100)
    output: Output = Field(default_factory=Output)
    captions: Captions = Field(default_factory=Captions)
    audio: Audio = Field(default_factory=Audio)

    @model_validator(mode='after')
    def bounds(self):
        if len({s.id for s in self.segments}) != len(self.segments):
            raise ValueError('Segment identifiers must be unique')
        if sum(s.end-s.start for s in self.segments) > 7200:
            raise ValueError('Output cannot exceed two hours')
        return self

    def validate_source(self,duration):
        if any(s.end > duration+.001 for s in self.segments):
            raise ValueError('A segment extends beyond the source')
        if any(w.end > duration+.001 for w in self.captions.words):
            raise ValueError('A caption extends beyond the source')
        if any(k.time > duration+.001 for s in self.segments for k in s.crop.keyframes):
            raise ValueError('A crop keyframe extends beyond the source')
        return self

class TimelineUpdate(Strict):
    revision: int = Field(ge=1)
    timeline: Timeline

class Interval(Strict):
    start: float = Field(ge=0,le=86400)
    end: float = Field(gt=0,le=86400)
    @model_validator(mode='after')
    def valid(self):
        if self.end<=self.start: raise ValueError('Interval end must follow start')
        return self

class TranscriptionOptions(Strict):
    model: Literal['tiny.en','tiny'] = 'tiny.en'
    language: Literal['en','pt','auto'] = 'en'
    audio_stream: int = Field(default=0,ge=0,le=31)

    @model_validator(mode='after')
    def supported_language(self):
        if self.model == 'tiny.en' and self.language != 'en':
            raise ValueError('tiny.en supports English only; select tiny for Portuguese or language detection')
        return self

class AnalysisOptions(TranscriptionOptions):
    genre: Literal['speech','tutorial','action','mixed'] = 'speech'
    mode: Literal['local','heuristic','agent','cloud'] = 'local'
    query: str = Field(default='',max_length=2000)
    min_length: float = Field(default=5,ge=1,le=300)
    max_length: float = Field(default=30,ge=1,le=300)
    target_count: int = Field(default=5,ge=1,le=30)
    min_separation: float = Field(default=2,ge=0,le=300)
    exclusions: list[Interval] = Field(default_factory=list,max_length=100)
    transcribe: bool = True
    use_feedback: bool = False
    visual_search: bool = False
    visual_model: Literal['clip-vit-b32','xclip-b16-k600'] = 'clip-vit-b32'
    impact_detection: bool = False
    impact_low_hz: float = Field(default=1000,ge=0,le=7999)
    impact_high_hz: float = Field(default=6000,ge=1,le=8000)
    impact_threshold: float = Field(default=.04,ge=.000001,le=1000)
    impact_min_gap: float = Field(default=.6,ge=.01,le=300)
    impact_lead: float = Field(default=.4,ge=0,le=300)
    impact_follow: float = Field(default=.5,ge=0,le=300)
    @model_validator(mode='after')
    def lengths(self):
        if self.max_length < self.min_length: raise ValueError('Maximum length must be at least minimum')
        if self.visual_search and (self.mode != 'local' or not self.query.strip()):
            raise ValueError('Visual search requires local mode and an English description of visible content.')
        if self.impact_detection:
            if self.visual_search or self.genre not in ('action','mixed') or self.mode not in ('local','heuristic'):
                raise ValueError('Impact detection requires action/mixed local or heuristic mode, without visual search.')
            from .impacts import _configuration
            _configuration(self.model_dump())
        return self

class OCROptions(Strict):
    ocr_model: Literal['ppocr-v4','ppocr-v5-mobile'] = 'ppocr-v4'
    ocr_mode: Literal['baseline','detail'] = 'baseline'
    ocr_quality_samples: int = Field(default=1, ge=1, le=3, strict=True)
    ocr_sample_seconds: float = Field(default=5, ge=1, le=600)
    ocr_max_samples: int = Field(default=120, ge=2, le=120)

class ProjectPatch(Strict):
    title: str | None = Field(default=None,min_length=1,max_length=200)
    tags: list[str] | None = Field(default=None,max_length=20)
    genre: Literal['speech','tutorial','action','mixed'] | None = None
    attribution: str | None = Field(default=None,max_length=2000)
    permission_status: Literal['unknown','permission-recorded','owned','licensed','synthetic'] | None = None

class Candidate(Strict):
    id: str = Field(min_length=1,max_length=80)
    start: float = Field(ge=0,le=86400)
    end: float = Field(gt=0,le=86400)
    title: str = Field(min_length=1,max_length=200)
    rationale: str = Field(max_length=3000)
    evidence: list[dict] = Field(default_factory=list,max_length=100)
    confidence: float = Field(ge=0,le=1)
    uncertainty: str = Field(max_length=3000)
    scores: dict[str,float] = Field(default_factory=dict)
    method: str = Field(max_length=100)
    @model_validator(mode='after')
    def bounded(self):
        if self.end<=self.start: raise ValueError('Invalid candidate times')
        if any(not math.isfinite(v) for v in self.scores.values()): raise ValueError('Nonfinite score')
        return self

class AgentDecisions(Strict):
    schema_version: Literal[1] = 1
    packet_fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')
    agent_name: str = Field(min_length=1,max_length=80)
    candidates: list[Candidate] = Field(max_length=30)

class PublicationDraft(Strict):
    project_id: str
    export_id: str
    destination: Literal['manual','youtube','tiktok','instagram'] = 'manual'
    account: str = Field(default='',max_length=200)
    caption: str = Field(default='',max_length=5000)
    visibility: Literal['private','unlisted','public'] = 'private'
    schedule: str | None = None
    timezone: str = 'Australia/Perth'
    permission_status: str = Field(default='unknown',max_length=200)
    @model_validator(mode='after')
    def schedule_valid(self):
        from zoneinfo import ZoneInfo,ZoneInfoNotFoundError
        from datetime import datetime
        try:ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError,ValueError):raise ValueError('Use a valid IANA time zone') from None
        if self.schedule:
            try:datetime.fromisoformat(self.schedule)
            except ValueError:raise ValueError('Schedule must be an ISO 8601 date and time') from None
        return self

def default_timeline(duration: float) -> dict:
    return Timeline(segments=[Segment(id='s1',start=0,end=min(duration,30))]).model_dump()
