# Security scope

Vektri Clips is a local, single-operator application. Keep the API on `127.0.0.1`. Host/origin validation and filesystem checks do not replace authentication. Do not expose it through a public reverse proxy or share the local data directory.

Imported media, transcripts, model outputs and external proposals are untrusted input. Keep dependencies current, preserve originals and back up private runtime data before upgrading. No credentials, recordings, database, model weights or private diagnostic reports belong in this repository.

If you suspect a vulnerability, do not post credentials, media or exploit details in a public issue. Contact the repository owner privately first. This document does not promise a response time, bug bounty or security certification.
