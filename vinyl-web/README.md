# Vinyl scanner (browser)

Static page that matches a webcam frame, or a still image, against the local cover index. OpenCV.js runs ORB (1000 features, longest side 800px) on the center viewfinder only. Matching is brute-force Hamming with Lowe's ratio 0.75. A hit is kept only when `findHomography` + RANSAC (5px) returns a valid quadrilateral and either confidence is above 85% or there are more than 20 inliers.

The camera loop does not call the network. OpenCV.js is loaded once from the OpenCV CDN when the page opens. `database.json`, `wantlist.json`, and `covers/` are local. Discogs is contacted only when you press **Дані Discogs** or **Оновити з Discogs** in crate mode.

## Open

`getUserMedia` and OpenCV.js need http, not `file://`.

From this directory:

```bash
python3 -m http.server
```

Then open http://127.0.0.1:8000/

Allow the camera if the browser asks. A green viewfinder marks the region that is matched. **Стоп** stops the camera.

If the camera is denied, use **Зображення** to pick a still, or **Обкладинка Ultra** to run the bundled Depeche Mode cover. That button is enough to prove a match without a webcam.

## Modes

The switcher is on the main screen.

- **Разова перевірка** (Single Check). One cover at a time. The page runs the full homography match and replaces the overlay. It does not add to the session.
- **Копання ящиків** (Crate Digging). The same matcher keeps running. When a hit's album id was not added in the last 5 seconds, the page appends it to a session file kept in memory, shows a short toast, and calls `navigator.vibrate` when the browser allows it. A repeat of that id inside 5 seconds is ignored. Refreshing the page clears the session.

The overlay shows artist, album, year, primary style, a related hint (`Related: …`), and ownership. This database has no ownership fields, so the line is `ownership unknown`. A Discogs price line is drawn only when the record has a numeric or text `discogs_price` or `price`. No price is invented, and this database has neither field.

Metadata stays as it is in `database.json`. Buttons are in Ukrainian.

## Crate file

The session is one object, not a bare array. `created` is the ISO timestamp of the first accepted row. Each item starts with `discogs: null` and `on_wishlist: null`.

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
      "on_wishlist": null
    }
  ]
}
```

`confidence` is the inlier ratio from 0 to 1. After the first row, three buttons appear under **Копання ящиків**:

1. **Дані Discogs.** For each item, one request to `https://api.discogs.com/database/search?q=ARTIST+ALBUM&type=release`, with a descriptive `User-Agent`. No token. The first result is stored as `discogs`: id, title, year, label (the first label name), country, and a `https://www.discogs.com` uri. `community` (have/want) is copied only when that result includes it. `price` is copied only from `price` or `lowest_price` on that result. Nothing is invented. A failed request, including an empty result list, leaves `discogs` null and shows the error on that row. This does not run during the camera loop.
2. **Список бажань.** The username field loads as `Look_at_my_scarface`. This button matches rows against the shipped `wantlist.json` (68 releases, each `id`, `artist`, and `title` only) with no token, so it works offline. **Оновити з Discogs** reloads `https://api.discogs.com/users/{username}/wants` when online, with a descriptive `User-Agent` and no token. A row is так when its Discogs release id is in the list, or when the artist and album match, ignoring case. Otherwise it is ні. A private list still shows the private-list message and does not change the rows.
3. **Експорт для ШІ.** Downloads `crate_session_YYYY-MM-DD.json` (the object above, with whatever Discogs and wantlist fields are filled in) and `crate_session_YYYY-MM-DD.txt`. The text file is one short block per record: artist, album, year, genre, styles, confidence, discogs url, on_wishlist. Paste it into a chat. The download does not include the username or the token.

## Covers

- `covers/depeche_mode_ultra.jpg` is the real Ultra cover. Do not replace it.
- `covers/massive_attack_mezzanine.jpg` and `covers/portishead_dummy.jpg` are synthetic stand-ins, copied as-is. They match these images, not photographs of the real sleeves.

## Limits

The scan loop does not call a backend, Discogs, or Spotify. Discogs runs only from **Дані Discogs** and **Оновити з Discogs**. A sleeve has to sit inside the green frame, and it has to be one of the indexed images. Steep angles, blur, and glare miss. Ultra matches this supplied file, not a different photo of the same sleeve. The Discogs search returns whatever release ranks first for that artist and album. It is not guaranteed to be the same pressing as the local year.
