import imghdr
import numpy as np
import json
import io
from starlette.responses import StreamingResponse
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses
import math
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.framework.formats import (detection_pb2, landmark_pb2,location_data_pb2)
from tensorflow import keras

store_faces = []
store_mesh = []
store_coordinates = []
model=load_model("./final_model.h5")
#https://upload-aws-hack.s3.amazonaws.com/final_model.h5
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_RGB_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
bg_img=0

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)

@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = BLUE_COLOR 
    thickness: int=2
    circle_radius: int=2

def load_img(path):
    image = cv2.resize(path, (224, 224))
    return image[...,::-1]


def prepare(proxy_image):
    IMG_SIZE = 224
    new_array = cv2.resize(proxy_image, (IMG_SIZE, IMG_SIZE)) 
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE,3)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    
    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and 
    is_valid_normalized_value(normalized_y)): return None
    # TODO: Draw coordinates even if it's outside of the image bounds.
    
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def draw_detection(
    image: np.ndarray,
    detection: detection_pb2.Detection,
    keypoint_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    bbox_drawing_spec: DrawingSpec = DrawingSpec()):

    """Draws the detction bounding box and keypoints on the image.

    Args:
    image: A three channel RGB image represented as numpy ndarray.
    detection: A detection proto message to be annotated on the image.
    keypoint_drawing_spec: A DrawingSpec object that specifies the keypoints'
        drawing settings such as color, line thickness, and circle radius.
    bbox_drawing_spec: A DrawingSpec object that specifies the bounding box's 
        drawing settings such as color and line thickness.

    Raises:
    ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If the location data is not relative data.
    """

    if not detection.location_data: return
    if image.shape[2] != _RGB_CHANNELS: 
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape

    location = detection.location_data
    if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
        raise ValueError(
            'LocationData must be relative for this drawing funtion to work.')

    # Draws keypoints.
    for keypoint in location.relative_keypoints:
        keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                        image_cols, image_rows)
        cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
                    keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)

    # Draws bounding box if exists.
    if not location.HasField('relative_bounding_box'): return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    cv2.rectangle(image, rect_start_point, rect_end_point,
                bbox_drawing_spec.color, bbox_drawing_spec.thickness)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec,
                                 Mapping[int, DrawingSpec]] = DrawingSpec(
                                     color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec,
                                   Mapping[Tuple[int, int],
                                           DrawingSpec]] = DrawingSpec()):
    
    """Draws the landmarks and the connections on the image.

    Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on 
        the image.
    connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
    landmark_drawing_spec: Either a DrawingSpec object or a mapping from
        hand landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius.
        If this argument is explicitly set to None, no landmarks will be drawn.
    connection_drawing_spec: Either a DrawingSpec object or a mapping from
        hand connections to the DrawingSpecs that specifies the
        connections' drawing settings such as color and line thickness.
        If this argument is explicitly set to None, no landmark connections will
        be drawn.

    Raises:
    ValueError: If one of the followings:
        a) If the input image is not three channel RGB.
        b) If any connetions contain invalid landmark index.
    """

    if not landmark_list: return
    if image.shape[2] != _RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and 
        landmark.visibility < _VISIBILITY_THRESHOLD) or 
        (landmark.HasField('presence') and 
        landmark.presence < _PRESENCE_THRESHOLD)): continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                    image_cols, image_rows)
        if landmark_px: idx_to_coordinates[idx] = landmark_px
    if connections: 
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(
                    connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx],
                            idx_to_coordinates[end_idx], drawing_spec.color,
                            drawing_spec.thickness)
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(
                landmark_drawing_spec, Mapping) else landmark_drawing_spec
            # White circle border
            circle_border_radius = max(drawing_spec.circle_radius + 1,
                                        int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR,
                        drawing_spec.thickness)
            # Fill color into the circle
            cv2.circle(image, landmark_px, drawing_spec.circle_radius,
                        drawing_spec.color, drawing_spec.thickness)


def draw_axis(
    image: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    focal_length: Tuple[float, float] = (1.0, 1.0),
    principal_point: Tuple[float, float] = (0.0, 0.0),
    axis_length: float = 0.1,
    axis_drawing_spec: DrawingSpec = DrawingSpec()):
    
    """Draws the 3D axis on the image.

    Args:
    image: A three channel RGB image represented as numpy ndarray.
    rotation: Rotation matrix from object to camera coordinate frame.
    translation: Translation vector from object to camera coordinate frame.
    focal_length: camera focal length along x and y directions.
    principal_point: camera principal point in x and y.
    axis_length: length of the axis in the drawing.
    axis_drawing_spec: A DrawingSpec object that specifies the xyz axis 
        drawing settings such as line thickness.

    Raises:
    ValueError: If one of the followings: 
        a) If the input image is not three channel RGB.
    """

    if image.shape[2] != _RGB_CHANNELS: raise ValueError(
        'Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    # Create axis points in camera coordinate frame.
    axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
    x = axis_cam[..., 0]
    y = axis_cam[..., 1]
    z = axis_cam[..., 2]
    # Project 3D points to NDC space.
    fx, fy = focal_length
    px, py = principal_point
    x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
    y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
    # Convert from NDC space to image space.
    x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
    y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
    # Draw xyz axis on the image.
    origin = (x_im[0], y_im[0])
    x_axis = (x_im[1], y_im[1])
    y_axis = (x_im[2], y_im[2])
    z_axis = (x_im[3], y_im[3])
    cv2.arrowedLine(image, origin, x_axis, RED_COLOR, axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, y_axis, GREEN_COLOR,
                    axis_drawing_spec.thickness)
    cv2.arrowedLine(image, origin, z_axis, BLUE_COLOR,
                    axis_drawing_spec.thickness)


def _normalize_color(color): return tuple(v / 255. for v in color)


def plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList, 
                    connections: Optional[List[Tuple[int, int]]] = None, 
                    landmark_drawing_spec: DrawingSpec = DrawingSpec(
                        color=RED_COLOR, thickness=5), 
                        connection_drawing_spec: DrawingSpec = DrawingSpec(
                            color=BLACK_COLOR, thickness=5),
                            elevation: int = 10,
                            azimuth: int = 10):

    """Plot the landmarks and the connections in matplotlib 3d.

    Args:
    landmark_list: A normalized landmark list proto message to be plotted.
        connections: A list of landmark index tuples that specifies how landmarks to
        be connected.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
        drawing settings such as color and line thickness.
        connection_drawing_spec: A DrawingSpec object that specifies the
        connections' drawing settings such as color and line thickness.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.

    Raises:
    ValueError: If any connetions contain invalid landmark index.
    """

    if not landmark_list: return
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
            landmark.visibility < _VISIBILITY_THRESHOLD) or
            (landmark.HasField('presence') and
            landmark.presence < _PRESENCE_THRESHOLD)): continue
        ax.scatter3D(
            xs=[-landmark.z],
            ys=[landmark.x],
            zs=[-landmark.y],
            color=_normalize_color(landmark_drawing_spec.color[::-1]),
            linewidth=landmark_drawing_spec.thickness)
        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection '
                                f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=_normalize_color(connection_drawing_spec.color[::-1]),
                    linewidth=connection_drawing_spec.thickness)
    plt.show()

@app.get("/")
async def root():
    return {"message": "Deep-Fake Detector"}


@app.post("/predict")
async def root(file: UploadFile = File(...)):
    global model
    global store_coordinates
    global store_faces
    global store_mesh
    global bg_img

    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    bg_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    if(bg_img is not None): gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    
    for (x, y, w, h) in faces:
        a = x-round(w/6)
        b = y-round(h/6)
        c = round((4*w)/3)
        d = round((4*h)/3)
        cropped_image = bg_img[b:b + d, a:a + c]
        temp_list = [a, b, c, d]
        store_faces.append(cropped_image)
        store_coordinates.append(temp_list)

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec=mp_drawing.DrawingSpec(color=BLUE_COLOR, thickness=1, circle_radius=1)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:

        for image in store_faces:
            prediction = model.predict(prepare(image.astype(np.float32)))
            if (prediction[0][0]>prediction[0][1]):
                if(image is not None): results = face_mesh.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks: print("Continue")
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        # connections=mp_face_mesh.FACEMESH_CONTOURS, 
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    store_mesh.append(annotated_image)
    if not store_mesh: pass
    else: 
        for i in range(len(store_coordinates)):
            x, y, w, h = store_coordinates[i]
            bg_img [b:b + d, a:a + c] = store_mesh[i]
    
    res,im_png = cv2.imencode(".png", bg_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    
@app.get("/predict")
async def resp():
    global bg_img
    res,im_png = cv2.imencode(".png", bg_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    #test=img.astype(np.float32)
    #prediction = model.predict(prepare(test))
    #return json.dumps(prediction.tolist())

    
