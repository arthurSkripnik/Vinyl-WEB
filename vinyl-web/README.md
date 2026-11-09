# Vinyl scanner (browser)

Static page that matches a webcam frame, or a still image, against the local cover index. OpenCV.js runs ORB (1000 features, longest side 800px) on the center viewfinder only. Matching is brute-force Hamming with Lowe's ratio 0.75. A hit is kept only when `findHomography` + RANSAC (5px) returns a valid quadrilateral and either confidence is above 85% or there are more than 20 inliers.

Nothing is fetched during the scan loop. OpenCV.js is loaded once from the OpenCV CDN when the page opens. `database.json` and `covers/` are local.

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
- **Копання ящиків** (Crate Digging). The same matcher keeps running. When a hit's album id was not added in the last 5 seconds, the page appends it to the session, shows a short toast, and calls `navigator.vibrate` when the browser allows it. A repeat of that id inside 5 seconds is ignored.

The overlay shows artist, album, year, primary style, a related hint (`Related: …`), and ownership. This database has no ownership fields, so the line is `ownership unknown`. A Discogs price line is drawn only when the record has a numeric or text `discogs_price` or `price`. No price is invented, and this database has neither field.

Metadata stays as it is in `database.json`. Buttons are in Ukrainian.

## Export

**Експорт сесії** downloads `crate_session_YYYY-MM-DD.json` (local date). The file is a JSON array. Each object has:

- `artist`
- `album`
- `year`
- `genre`
- `styles` (array of strings)
- `confidence` (inlier ratio from 0 to 1, four decimal places)

The button stays disabled until the crate session has at least one row. Switching back to single-check keeps the session in memory so it can still be exported. Refreshing the page clears it.

## Covers

- `covers/depeche_mode_ultra.jpg` is the real Ultra cover. Do not replace it.
- `covers/massive_attack_mezzanine.jpg` and `covers/portishead_dummy.jpg` are synthetic stand-ins, copied as-is. They match these images, not photographs of the real sleeves.

## Limits

The scan loop does not call a backend, Discogs, or Spotify. A sleeve has to sit inside the green frame, and it has to be one of the indexed images. Steep angles, blur, and glare miss. Ultra matches this supplied file, not a different photo of the same sleeve.
