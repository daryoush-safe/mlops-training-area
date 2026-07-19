from __future__ import annotations

from dataclasses import dataclass

from sqlgen.clients.config import ClientsConfig
from sqlgen.clients.database import DatabaseClient
from sqlgen.clients.vllm import VLLMChatClient
from sqlgen.config import Params, load_params


@dataclass
class Deps:
    params: Params
    pruner: VLLMChatClient
    sql_generator: VLLMChatClient
    presenter: VLLMChatClient
    db_client: DatabaseClient

    @classmethod
    def build(cls, params: Params | None = None, clients: ClientsConfig | None = None) -> "Deps":
        params = params or load_params()
        clients = clients or ClientsConfig.from_env()
        return cls(
            params=params,
            pruner=VLLMChatClient(
                clients.pruner.base_url, clients.pruner.model, clients.request_timeout
            ),
            sql_generator=VLLMChatClient(
                clients.sqlgen.base_url, clients.sqlgen.model, clients.request_timeout
            ),
            presenter=VLLMChatClient(
                clients.presenter.base_url, clients.presenter.model, clients.request_timeout
            ),
            db_client=DatabaseClient(clients.db_base_url, clients.db_timeout),
        )
