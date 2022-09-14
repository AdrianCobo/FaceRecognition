# Copyright 2022 Adrian Cobo Merino
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import face_recognition


correct_detection_color = (0, 255, 0)
incorrect_detection_color = (0, 0, 255)


class FaceRecognitor():

    def compare_faces(self, img1, img2, draw_bbx=False):
        results = []
        face_loc_img1 = face_recognition.face_locations(img1)[0]
        face_image_encodings_1 = face_recognition.face_encodings(img1, known_face_locations=[
            face_loc_img1])[0]  # da un vector de 128 caracteristicas para cada rostro
        face_locs_img2 = face_recognition.face_locations(img2)
        if face_locs_img2 != []:
            for face_location_img2 in face_locs_img2:
                face_img2_encodings = face_recognition.face_encodings(
                    img2, known_face_locations=[face_location_img2])[0]
                result = face_recognition.compare_faces(
                    [face_image_encodings_1], face_img2_encodings)[0]

                if draw_bbx == True:
                    bbx_color = correct_detection_color
                    if result == False:
                        bbx_color = incorrect_detection_color
                    cv2.rectangle(img2, (face_location_img2[3], face_location_img2[0]),
                                  (face_location_img2[1], face_location_img2[2]), bbx_color)
                    cv2.imshow("Image1 faces", img1)
                    cv2.imshow("Image2 faces", img2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                results.append(result)

        return results


def main():
    img_objective = cv2.imread("Images/Cobichuelo.jpeg")
    img_test1 = cv2.imread("Images/Cobichuelo2.jpg")
    img_test2 = cv2.imread("Images/Reynolds.jpeg")

    face_recognitor = FaceRecognitor()
    print(face_recognitor.compare_faces(img_objective, img_test1, True))
    print(face_recognitor.compare_faces(img_objective, img_test2, True))


if __name__ == "__main__":
    main()
