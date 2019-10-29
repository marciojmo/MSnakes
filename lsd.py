import cv2


class LSD:
    def __init__(self, image):
        self.lsd = cv2.createLineSegmentDetector(0)
        self.image = image
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.lines = None

    def detect_lines(self):
        lsd = cv2.createLineSegmentDetector(0)
        self.lines = lsd.detect(self.gray)[0]

    def visualize(self):
        lines_image = self.lsd.drawSegments(self.image, self.lines)
        cv2.imshow("Line Segments", lines_image)
        cv2.waitKey(0)

