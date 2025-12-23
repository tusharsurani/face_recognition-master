import face_recognition
import cv2

input_movie = cv2.VideoCapture("hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Lin-Manuel Miranda"
        elif match[1]:
            name = "Alex Lacamoire"

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
