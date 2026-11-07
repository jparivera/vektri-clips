"""Typed extension boundaries. An interface is not a configured remote integration."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol,Callable
from .security import fetch_public_media

class IntegrationUnavailable(RuntimeError):pass

class SourceAdapter(Protocol):
    name:str
    def supports(self,url:str)->bool: ...
    def fetch(self,url:str,destination:Path,check:Callable[[],None])->dict: ...

class PublicMediaAdapter:
    name='direct-public-media-v1'
    def supports(self,url):return url.startswith(('https://','http://'))
    def fetch(self,url,destination,check):return fetch_public_media(url,destination,check)

@dataclass(frozen=True)
class PublishingCapabilities:
    configured:bool
    can_upload:bool
    can_reconcile:bool
    prerequisites:tuple[str,...]

class PublishingAdapter(Protocol):
    """Future adapters must persist operation IDs before retries and reconcile uncertainty."""
    name:str
    def capabilities(self)->PublishingCapabilities: ...
    def validate_destination(self,account:str)->dict: ...
    def upload(self,asset:Path,approved_draft:dict,operation_key:str,check:Callable[[],None])->dict: ...
    def reconcile(self,operation_key:str,remote_id:str|None)->dict: ...

class ManualPublishingAdapter:
    name='manual'
    def capabilities(self):return PublishingCapabilities(True,False,False,('Operator uploads the downloaded package using the destination platform.',))
    def validate_destination(self,account):return {'account':account,'verified':False,'method':'operator-entered manual destination'}
    def upload(self,*args,**kwargs):raise IntegrationUnavailable('Manual handoff cannot upload or publish. No remote operation was attempted.')
    def reconcile(self,*args,**kwargs):raise IntegrationUnavailable('No remote operation exists to reconcile for a manual handoff.')

@dataclass(frozen=True)
class CloudEstimate:
    provider:str
    model:str
    media_bytes:int
    estimated_max_usd:float
    evidence_fingerprint:str
    transmitted_fields:tuple[str,...]

class CostGovernedCloudAnalysis(Protocol):
    """No implementation is registered. Cloud requests reject before network access.

    An implementation must add transactional spend reservations, hard caps and explicit transfer approval.
    """
    def estimate(self,packet:dict)->CloudEstimate: ...
    def analyze(self,packet:dict,approved_estimate:CloudEstimate,budget_reservation_id:str,check:Callable[[],None])->dict: ...
