# Publishing boundary and current prerequisites

Delivery behavior: export a verified file/package, prepare a draft, review it and mark it ready for manual upload. No platform account, API key or posting transport is connected. There were zero paid API calls and zero posts. The `/publish` route returns 409; a draft adapter is not represented as a working platform integration.

A draft records project, exact export/revision/fingerprint, destination, account, caption, visibility, schedule, IANA time zone and user-recorded permission status. Approval repeats the complete draft and asset fingerprint plus a meaningful rights acknowledgment and `confirm_manual_handoff:true`. Approval is scoped to manual handoff. Timeline, transcript, provenance or draft edits invalidate it. A schedule is metadata, not an active scheduled task. A checkbox or recorded acknowledgment does not establish rights to imported material.

## Official references checked 2026-11-03

| Provider | Reviewed official interface | External prerequisites and implementation status |
|---|---|---|
| YouTube | [Upload guide](https://developers.google.com/youtube/data/guides/uploading_a_video), [videos.insert](https://developers.google.com/youtube/data/reference/rest/v3/videos/insert) | Google Cloud project with YouTube Data API, OAuth configuration and authorized user, appropriate upload scope, current quota/verification eligibility review. No credentials or transport implemented. Do not rely on old quota/private-upload claims without checking current docs. |
| TikTok | [Content Posting API getting started](https://developers.tiktok.com/docs/en/content-posting-api-get-started) (page version updated 2026-08-04) | Registered app, Content Posting/Direct Post configuration, approved `video.publish` scope and authorized user. Unaudited clients have visibility restrictions. Creator info and current capabilities must be fetched; direct URL pulls require verified domains, or use supported file upload. No account/transport implemented. |
| Instagram | [Content Publishing](https://developers.facebook.com/docs/instagram-platform/content-publishing/) | Official documentation fetch returned HTTP 429 in this environment. Current account eligibility, scopes, review requirements and upload behavior remain unverified. No integration attempted; use manual upload. |

Before a future real adapter can be enabled: implement the official supported upload/status interface; persist operation intent and remote IDs; bind approval to final asset and destination; reconcile uncertain responses before retry; validate current quotas, scopes and account eligibility; test duplicate prevention against the real provider. Never assume an import authorizes publishing. Deletion/takedown/account actions require separate explicit authority. Analytics must come from real provider responses, never generated sample engagement.

The current `PublishingAdapter` protocol and `ManualPublishingAdapter` define the seam. `CloudAnalysisAdapter` likewise defines an estimate/approval/budget seam but has no registered paid provider. Separate API credentials and reviewed cost/data-transfer controls are required; ChatGPT/Claude subscriptions are not treated as API credit. Work on local import/edit/AI/render can continue without these external connections.
