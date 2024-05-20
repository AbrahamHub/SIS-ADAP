import threading
import time
import random

class TrafficLight:
    def __init__(self, name):
        self.name = name
        self.green = True

    def switch_light(self):
        self.green = not self.green

    def get_state(self):
        return "Green" if self.green else "Red"

class VehicleDetector:
    def __init__(self):
        self.vehicle_count = 0

    def detect_vehicle(self):
        self.vehicle_count += 1

    def get_vehicle_count(self):
        return self.vehicle_count
class TrafficController:
    def __init__(self, traffic_lights):
        self.traffic_lights = traffic_lights

    def control_traffic(self):
        while True:
            for light in self.traffic_lights:
                if light.get_state() == "Green":
                    print(f"{light.name} light is Green. Allowing traffic...")
                else:
                    print(f"{light.name} light is Red. Stopping traffic...")
            time.sleep(5)
            self.switch_lights()

    def switch_lights(self):
        for light in self.traffic_lights:
            light.switch_light()

def traffic_simulation():
    north_south_light = TrafficLight("North-South")
    east_west_light = TrafficLight("East-West")

    detector = VehicleDetector()

    controller = TrafficController([north_south_light, east_west_light])

    controller_thread = threading.Thread(target=controller.control_traffic)
    detector_thread = threading.Thread(target=simulate_vehicle_detection, args=(detector,))

    controller_thread.start()
    detector_thread.start()

    controller_thread.join()
    detector_thread.join()

def simulate_vehicle_detection(detector):
    while True:
        time.sleep(random.randint(1, 3))
        detector.detect_vehicle()
        print(f"Vehicle detected. Total count: {detector.get_vehicle_count()}")

if __name__ == "__main__":
    traffic_simulation()
