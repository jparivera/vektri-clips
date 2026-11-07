# Frontend dependency and asset notes

Reviewed 2026-11-03 against current npm package metadata, installed license files, [React documentation](https://react.dev/reference/react), and [Vite documentation](https://vite.dev/guide/). All direct dependencies have exact versions in `package.json`, with integrity-pinned transitive versions in `package-lock.json`.

| Package | Version | License | Role |
|---|---|---|---|
| React / React DOM | 19.3.1 | MIT | Browser runtime |
| Scheduler | 0.28.0 | MIT | React runtime dependency |
| Vite | 8.3.5 | MIT | Development/build tooling |
| TypeScript | 7.0.2 | Apache-2.0 | Type checking |
| React types | 19.2.14 | MIT | Type declarations |
| React DOM types | 19.2.3 | MIT | Type declarations |
| Playwright | 1.64.1 | Apache-2.0 | Local browser testing |
| Prettier | 3.9.10 | MIT | Source formatting |

Runtime license texts are preserved in `public/third-party-notices.txt` and copied into the built application. No proprietary logos, product code, downloaded web fonts, icon packs, analytics or external UI resources are included. The Vektri word treatment and simple geometric UI icons are authored for this project. Browser typography uses local/system fonts; rendered captions use the separately audited system DejaVu font.

`../docs/licenses/javascript-inventory.json` records all lockfile packages and distinguishes installed dependencies from optional binaries for other platforms. Installed tooling dependencies include Lightning CSS (MPL-2.0), Picocolors (ISC), Source Map JS (BSD-3-Clause), and MIT/Apache-2.0 dependencies. Development tools are not included in the browser runtime bundle. This does not license a future redistribution of a whole `node_modules` directory or Chrome binary; retain their own notices and audit the actual package if doing so. Optional platform packages not installed on this Linux machine are explicitly marked unreviewed in the inventory.
