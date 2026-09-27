from pydantic import BaseModel

from gateway.models.workers.served_model import ServedModel

class ModelList(BaseModel):
    data: list[ServedModel]

    def contains(self, model_name: str) -> bool:
        return any(model.id == model_name for model in self.data)
