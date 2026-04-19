from app.repositories.plan_repo import PlanRepository


class PlanService:

    def __init__(self):
        self.repo = PlanRepository()

    def get_all_plans(self, db):
        return self.repo.get_all(db)

    def get_plan(self, db, plan_id: int):
        plan = self.repo.get_by_id(db, plan_id)
        if not plan:
            raise Exception("Plan not found")
        return plan

    def validate_plan_access(self, plan, usage: dict):
        """
        usage example:
        {
            "repos": 3,
            "requests": 120
        }
        """

        if usage["repos"] > plan.max_repos:
            return False, "Repo limit exceeded"

        if usage["requests"] > plan.max_requests:
            return False, "Request limit exceeded"

        return True, "OK"