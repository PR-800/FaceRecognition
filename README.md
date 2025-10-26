# Face Recognition API

This repository contains a Flask-based API for advanced face recognition, incorporating liveness detection to prevent spoofing attacks and face matching against a known database.

## Features

-   **RESTful API:** Provides simple HTTP endpoints for face detection and verification.
-   **Face Detection:** Utilizes OpenCV's Haar Cascades to efficiently locate faces in images.
-   **Advanced Liveness Detection:** Employs a multi-feature analysis on the detected face to ensure the subject is a live person and not a photo or screen. This includes checks for:
    -   Blur (Laplacian Variance)
    -   Texture (Local Binary Patterns)
    -   Frequency Domain (FFT)
    -   Color Diversity (HSV)
    -   Edge Density (Canny)
    -   Brightness
-   **Face Matching:** Uses the `DeepFace` library with the `Facenet` model to generate facial embeddings and compare them against a database of known individuals using cosine similarity.
-   **Dynamic Database:** Automatically builds the face database at startup by processing images from the `dataset` folder.

## Project Structure

```
.
├── app.py              # Flask application, API endpoints, and main logic
├── model.py            # Core classes for face detection, liveness, and matching
├── requirements.txt    # Python dependencies
├── utils.py            # Utility functions for image processing
├── dataset/            # Directory for storing images of known individuals
└── dataset_for_test/   # Directory containing sample images for testing the API
```

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Clone the repository:**

```bash
git clone https://github.com/PR-800/FaceRecognition.git
cd FaceRecognition
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**

The `requirements.txt` file lists all necessary packages.

```bash
pip install -r requirements.txt
```

**4. Populate the Face Database:**

Add images of known individuals to the `dataset/` directory.

-   The file name (without extension) will be used as the person's identifier.
-   Use one clear image per person.
-   Supported formats: `.jpg`, `.png`, `.jpeg`.

For example:
-   `dataset/john_doe.jpg`
-   `dataset/jane_smith.png`

**5. Run the Application:**

```bash
python app.py
```

The API will start on `http://0.0.0.0:5000`. At startup, it will process the images in the `dataset` folder to build the face embeddings database.

## API Endpoints

### 1. Detect Face

This endpoint detects the largest face in an uploaded image and returns its coordinates and a cropped version of the face.

-   **Endpoint:** `POST /detect_face`
-   **Request:** `multipart/form-data` with an `image` file.

**Example `curl` Request:**

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://127.0.0.1:5000/detect_face
```

**Success Response (`200 OK`):**

```json
{
  "success": true,
  "message": "Face detected successfully",
  "face_coordinates": {
    "x1": 150,
    "y1": 210,
    "x2": 450,
    "y2": 550
  },
  "cropped_face_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Error Response (`400 Bad Request`):**

```json
{
  "success": false,
  "error": "No face detected in image"
}
```

### 2. Verify Face

This is the main endpoint that performs a two-step verification: liveness detection followed by face matching.

-   **Endpoint:** `POST /verify_face`
-   **Request:** `multipart/form-data` with an `image` file.

**Workflow:**
1.  **Liveness Detection:** The system first checks if the image contains a live person.
2.  **Face Matching:** If the liveness check is successful, the system proceeds to compare the face against the database. If the liveness check fails, this step is skipped.

**Example `curl` Request:**

```bash
curl -X POST -F "image=@/path/to/your/image.jpg" http://127.0.0.1:5000/verify_face
```

**Success Response (`200 OK`):**

The response indicates the results of both liveness and matching checks.

```json
{
  "is_live": true,
  "liveness_score": 0.85,
  "match_found": true,
  "matched_person": "john_doe",
  "similarity_score": 0.92
}
```

-   `is_live`: `true` if the liveness score is above the threshold (0.50).
-   `liveness_score`: A score from 0.0 to 1.0 indicating the probability of the subject being live.
-   `match_found`: `true` if a face in the database matches with a similarity score above the threshold (0.40).
-   `matched_person`: The identifier of the best-matched person from the database. `null` if no match is found.
-   `similarity_score`: The cosine similarity score (0.0 to 1.0) for the best match.