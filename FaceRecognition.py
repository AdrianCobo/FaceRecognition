# code from https://www.youtube.com/watch?v=51J_bYYMO2k
import cv2
import face_recognition

# Imagen a comparar
image = cv2.imread("Images/Cobichuelo.jpeg")
face_loc = face_recognition.face_locations(image)[0]
# print("face_loc:", face_loc) # ubicaccion de la cara
face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[
                                                       face_loc])[0]  # da un vector de 128 caracteristicas para cada rostro
print("face_image_encodings:", face_image_encodings)

# cv2.rectangle(image, (face_loc[3], face_loc[0]),
#               (face_loc[1], face_loc[2]), (0, 255, 0))
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

####################################################################################################################
# Probar si funciona la deteccion con un video

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if ret == False:
#         break
#     frame = cv2.flip(frame, 1)

#     face_locations = face_recognition.face_locations(frame)
#     if face_locations != []:
#         for face_location in face_locations:
#             face_frame_encodings = face_recognition.face_encodings(
#                 frame, known_face_locations=[face_location])[0]
#             result = face_recognition.compare_faces(
#                 [face_image_encodings], face_frame_encodings)
#             print(result)
#             cv2.rectangle(frame, (face_loc[3], face_loc[0]),
#                           (face_loc[1], face_loc[2]), (0, 255, 0))

#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == 27 & 0XFF:  # pulsar escape
#         break

# cap.release()
# cv2.destroyAllWindows()

####################################################################################################################
# Probar si funciona la deteccion con una imagen


def compare_faces_img(known_face_encoding, img2):
    face_locations = face_recognition.face_locations(img2)
    if face_locations != []:
        for face_location in face_locations:
            face_frame_encodings = face_recognition.face_encodings(
                img2, known_face_locations=[face_location])[0]
            result = face_recognition.compare_faces(
                [known_face_encoding], face_frame_encodings)
            print("Result:", result)


img_test1 = cv2.imread("Images/Cobichuelo2.jpg")
img_test2 = cv2.imread("Images/Reynolds.jpeg")
compare_faces_img(face_image_encodings, img_test1)
compare_faces_img(face_image_encodings, img_test2)
