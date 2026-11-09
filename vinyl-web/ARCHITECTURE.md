# Crate digging

Keep Single Check. In Копання ящиків:

1. Before matching, compute Laplacian variance on the viewfinder ROI. If the score is under 100, drop the frame and do not match.
2. Keep ORB, Lowe ratio 0.75, homography RANSAC 5.0, match only above 85% or more than 20 inliers. Keep the 5-second dedupe.
3. When a captured frame is sharp enough and does not match, save a compressed JPEG of the ROI as unknown_NN.jpg via download or an in-memory list the export includes, and add a row [UNKNOWN_COVER_NN.JPG]. The camera loop does not save frames. One **Зняти** (or one still) saves one unknown.
4. Barcode: if the browser BarcodeDetector exists, try it on an unmatched sharp frame and store the code. If it does not exist, skip barcode and still save the unknown image. Do not add a native dependency.
5. Export crate_session_YYYY-MM-DD.txt in this shape: header with the date, then [MATCHED IN LOCAL DB] lines as "N. Artist - Album (year) | Genre: ... | Status: ..." using wantlist (так means IN DISCOGS WANTLIST) and spotify_source when present, then [UNKNOWN / UNMATCHED] lines, then Total Scanned / Matched / Unknown. Keep the JSON export too.
6. Do not invent Spotify or friend-wantlist status. Do not publish.
