from locust import HttpUser, task, between, LoadTestShape


class AQIPredictorUser(HttpUser):
    wait_time = between(1, 2)

    # Based on Person 1's Lambda code, features must be a flat array
    payload_lgbm = {
        # 10 features
        "features": [45.2, 78.1, 32.5, 1.2, 8.3, 65.0, 22.1, 58.0, 3.2, 1013.0, 0.0]
    }

    payload_xgb = {
        # 12 features
        "features": [45.2, 78.1, 32.5, 1.2, 8.3, 65.0, 22.1, 58.0, 3.2, 1013.0, 0.0, 0.0]
    }

    @task(1)
    def test_xgb(self):
        self.client.post("/predict/xgb", json=self.payload_xgb)

    @task(1)
    def test_lgbm(self):
        self.client.post("/predict/lgbm", json=self.payload_lgbm)

# Custom shape to simulate the 4-Phase Burst


class BurstWorkload(LoadTestShape):
    stages = [
        # Phase 1: Baseline (5m)
        {"duration": 300, "users": 10, "spawn_rate": 2},
        # Phase 2: Ramp-up (2m)
        {"duration": 420, "users": 600, "spawn_rate": 10},
        # Phase 3: Sustain (5m)
        {"duration": 720, "users": 600, "spawn_rate": 10},
        # Phase 4: Recovery (5m)
        {"duration": 1020, "users": 10, "spawn_rate": 10},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
