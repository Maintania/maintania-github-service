from app.repositories.subscription_repo import SubscriptionRepository

class SubscriptionService:

    def __init__(self):
        self.repo = SubscriptionRepository()

    def get_user_subscription(self, db, user_id):
        return self.repo.get_by_user(db, user_id)

    def create_subscription(self, db, user_id, plan_id):
        return self.repo.create(db, {
            "user_id": user_id,
            "plan_id": plan_id
        })