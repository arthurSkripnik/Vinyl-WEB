# Vinyl scanner (browser)

Static page that matches a webcam frame, or a still image, against the local cover index. OpenCV.js runs ORB (1000 features, longest side 800px) on the center viewfinder only. Matching is brute-force Hamming with Lowe's ratio 0.75. A hit is kept only when `findHomography` + RANSAC (5px) returns a valid quadrilateral and either confidence is above 85% or there are more than 20 inliers.

The camera loop does not call the network. OpenCV.js is loaded once from the OpenCV CDN when the page opens. `database.json`, `wantlist.json`, and `covers/` are local. Discogs is contacted only when you press **Дані Discogs** or **Оновити з Discogs**. Gemini is contacted only after a sharp unknown cover is queued, and only when the key field on this page is not empty.

## Open

`getUserMedia` and OpenCV.js need http, not `file://`.

From this directory:

```bash
python3 -m http.server
```

Then open http://127.0.0.1:8000/

The camera stays off until **Камера**. **Зняти** captures one frame through the blur gate and the local match. **Стоп** stops the camera. A green viewfinder marks the region that is matched.

If the camera is denied, use **Зображення** to pick a still, or **Обкладинка Ultra** to run the bundled Depeche Mode cover. That button is enough to prove a match without a webcam.

## Screen

The page is one phone column, top to bottom:

1. **Ящик** — a selector to switch crates, plus a name field and **Створити** to add one. Crates stay in memory until reload.
2. A large viewfinder. The camera stays off until **Камера**. **Зняти** captures one frame. A frame is kept only when the viewfinder Laplacian variance is at least 100.
3. Live counts: **Скановано**, **Обробка**, **Збіги**.
4. **Вміст ящика**. Each row has a status: у локальній базі, у списку бажань when the baked wantlist matches, розпізнано Gemini, обробка, or невідомо.

A sharp local hit is ignored for 5 seconds if that album id was just added to the current crate. Each sharp miss from **Зняти** (or from a still) is stored once as `unknown_NN.jpg`. The camera loop does not queue misses. If the Gemini key field is empty, that row stays невідомо and no request is made. If a key is present, one cover is sent at a time, after the frame has been queued, not from inside the camera loop. The JPEG is scaled so its longest side is about 768px before that request. On 429 or 503 the queue pauses (30 seconds, or a small `Retry-After`) and then sends the same cover. The field is `gemini-api-key`. Its value is kept only in the browser's localStorage under `vinyl-scanner.gemini.apiKey` and is never written into this project, the page file, or the export. Refreshing the page clears the crates.

The overlay shows artist, album, year, primary style, a related hint (`Related: …`), and ownership. This database has no ownership fields, so the line is `ownership unknown`. A Discogs price line is drawn only when the record has a numeric or text `discogs_price` or `price`. No price is invented, and this database has neither field.

Metadata stays as it is in `database.json`. Buttons are in Ukrainian.

## Crate file

The export is the active crate, not a bare array. `created` is the ISO timestamp of the first accepted row. A local match is checked against the baked wantlist as soon as it is added. `on_wishlist` stays null until that list is loaded.

```json
{
  "created": "2026-11-09T20:00:00.000Z",
  "items": [
    {
      "id": "1",
      "artist": "Depeche Mode",
      "album": "Ultra",
      "year": 1997,
      "genre": "Electronic",
      "styles": ["Synth-pop", "Alternative Rock", "Dark Wave"],
      "confidence": 0.973,
      "discogs": null,
      "on_wishlist": null,
      "spotify_source": "Collection & Playlist: 90s Dark Wave"
    }
  ],
  "unknowns": []
}
```

`confidence` is the inlier ratio from 0 to 1. The crate section has these buttons:

1. **Дані Discogs.** For each item, one request to `https://api.discogs.com/database/search?q=ARTIST+ALBUM&type=release`, with a descriptive `User-Agent`. No token. The first result is stored as `discogs`: id, title, year, label (the first label name), country, and a `https://www.discogs.com` uri. `community` (have/want) is copied only when that result includes it. `price` is copied only from `price` or `lowest_price` on that result. Nothing is invented. A failed request, including an empty result list, leaves `discogs` null and shows the error on that row. This does not run during the camera loop.
2. **Список бажань.** The username field loads as `Look_at_my_scarface`. This button matches rows against the shipped `wantlist.json` (68 releases, each `id`, `artist`, and `title` only) with no token, so it works offline. **Оновити з Discogs** reloads `https://api.discogs.com/users/{username}/wants` when online, with a descriptive `User-Agent` and no token. A row is так when its Discogs release id is in the list, or when the artist and album match, ignoring case. Otherwise it is ні. A private list still shows the private-list message and does not change the rows.
3. **Експорт для ШІ.** Downloads `crate_session_YYYY-MM-DD.json` and `crate_session_YYYY-MM-DD.txt` for the crate that is selected. The text file has the date and crate name, `[MATCHED IN LOCAL DB]`, `[UNKNOWN / UNMATCHED]`, then Total Scanned, Matched, and Unknown. A matched status says `IN DISCOGS WANTLIST` only when the baked wantlist matched, and includes `spotify_source` only when that field is on the record. An enriched unknown line adds the artist, album, and year returned for that cover. The JSON keeps the matched items and includes each unknown JPEG. The Gemini key is not in either file. See `ARCHITECTURE.md`.

## Covers

- `covers/depeche_mode_ultra.jpg` is the real Ultra cover. Do not replace it.
- `covers/massive_attack_mezzanine.jpg` and `covers/portishead_dummy.jpg` are synthetic stand-ins, copied as-is. They match these images, not photographs of the real sleeves.

## Limits

The scan loop does not call a backend, Discogs, Spotify, or Gemini. Discogs runs only from **Дані Discogs** and **Оновити з Discogs**. Gemini runs only for a queued sharp unknown, and only when the key field is filled. A sleeve has to sit inside the green frame. The bundled Ultra JPEG scores under the blur gate, so this screen drops that still; Mezzanine and Dummy are sharp enough to match. The Discogs search returns whatever release ranks first for that artist and album. It is not guaranteed to be the same pressing as the local year.
