from locust import HttpUser, task, between, LoadTestShape


class AQIPredictorUser(HttpUser):
    wait_time = between(1, 2)

    payload_lgbm = {
        "features": [45.2, 78.1, 32.5, 1.2, 8.3, 65.0, 22.1, 58.0, 3.2, 1013.0, 0.0]
    }

    payload_xgb = {
        "features": [45.2, 78.1, 32.5, 1.2, 8.3, 65.0, 22.1, 58.0, 3.2, 1013.0, 0.0, 0.0]
    }

    # @task(1)
    def test_xgb_arpa(self):
        # 12 feature per ARPA
        self.client.post("/predict/xgb_arpa", json=self.payload_xgb)

    # @task(1)
    def test_xgb_baseline(self):
        # USA payload_lgbm CHE HA 11 FEATURE!
        self.client.post("/predict/xgb_baseline", json=self.payload_lgbm)

    # @task(1)
    def test_lgbm(self):
        # 11 feature per LGBM
        self.client.post("/predict/lgbm", json=self.payload_lgbm)

    @task(1)
    def test_lgbm_arpa(self):
        # 12 feature per ARPA
        self.client.post("/predict/xgb_arpa", json=self.payload_xgb)


'''
class BurstWorkload(LoadTestShape):
    stages = [
        # Phase 1: Baseline (5m) - 5 users
        {"duration": 300, "users": 5, "spawn_rate": 1},
        # Phase 2: Ramp-up (2m) - 150 users (Safe for AWS Academy!)
        {"duration": 420, "users": 150, "spawn_rate": 5},
        # Phase 3: Sustain (5m) - 150 users
        {"duration": 720, "users": 150, "spawn_rate": 5},
        # Phase 4: Recovery (5m) - 5 users
        {"duration": 1020, "users": 5, "spawn_rate": 5},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None
'''


class ShortWorkload(LoadTestShape):
    stages = [
        # Fase 1: Riscaldamento (primi 30 secondi) - 10 utenti
        {"duration": 30, "users": 10, "spawn_rate": 2},
        # Fase 2: Carico costante (fino a 5 minuti / 300 secondi) - 30 utenti
        {"duration": 300, "users": 50, "spawn_rate": 5},
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        # Il test si fermerà automaticamente dopo 300 secondi (5 minuti)
        return None


'''


class SpikeWorkload(LoadTestShape):
    def tick(self):
        run_time = self.get_run_time()
        # Instantly spawn 60 users and hold for 3 minutes
        if run_time < 180:
            return (60, 60)  # (Total Users, Spawn Rate per second)
        return None
'''
