"""The session end to end, driven from a gateway double over a real socket."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from conftest import (
    ALL_CAPABILITIES,
    CLAIM_CODE,
    RESUME_TOKEN,
    SESSION_ID,
    application,
    dial,
    entries,
    listening,
    members,
)
from tesseron import (
    ActionContext,
    DisconnectedEvent,
    DuplicateNameError,
    Emit,
    HandshakeFailedEvent,
    HostError,
    HostEvent,
    InvalidApplicationIdError,
    JsonObject,
    JsonValue,
    ManifestPublication,
    TesseronApp,
    Unsubscribe,
    WelcomeEvent,
)


class AddTodo(BaseModel):
    text: str = Field(min_length=1)
    tag: str | None = None


def todo_application() -> TesseronApp:
    app = application()

    @app.action("addTodo", description="Add one todo")
    async def add_todo(parsed: AddTodo, context: ActionContext) -> JsonValue:
        await context.progress(percent=100, message="saved")
        return {"text": parsed.text, "tag": parsed.tag}

    @app.action("wait", description="Block until cancelled")
    async def wait(raw_input: JsonValue, context: ActionContext) -> JsonValue:
        await context.cancellation.wait()
        return None

    async def read_cart() -> JsonValue:
        return {"total": 0}

    app.resource("cart", read=read_cart, description="Current cart", subscribable=True)
    return app


async def test_the_handshake_publishes_the_registered_manifest() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        hello = await gateway.receive()

        assert hello["method"] == "tesseron/hello"
        params = members(hello, "params")
        assert params["protocolVersion"] == "1.2.0"
        assert members(params, "app")["id"] == "testapp"
        assert params["capabilities"] == ALL_CAPABILITIES

        first_action = entries(params, "actions")[0]
        assert isinstance(first_action, dict)
        assert first_action["name"] == "addTodo"
        assert first_action["description"] == "Add one todo"
        schema = first_action["inputSchema"]
        assert isinstance(schema, dict)
        assert schema["required"] == ["text"]

        assert entries(params, "resources") == [
            {"name": "cart", "description": "Current cart", "subscribable": True}
        ]


async def test_an_invocation_streams_progress_then_answers_with_the_handler_output() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.invoke("addTodo", request_id="inv-1", input_value={"text": "buy milk"})

        progress = await gateway.receive()
        assert progress["method"] == "actions/progress"
        assert "id" not in progress
        assert members(progress, "params")["percent"] == 100

        answer = await gateway.receive()
        assert answer["id"] == "inv-1"
        assert members(answer, "result") == {
            "invocationId": "inv-1",
            "output": {"text": "buy milk", "tag": None},
        }


async def test_an_invocation_with_a_null_id_answers_with_a_null_id() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "id": None,
                "method": "actions/invoke",
                "params": {
                    "name": "addTodo",
                    "input": {"text": "buy milk"},
                    "invocationId": "null-request-id",
                },
            }
        )

        await gateway.receive()
        answer = await gateway.receive()
        assert answer["id"] is None
        assert members(answer, "result") == {
            "invocationId": "null-request-id",
            "output": {"text": "buy milk", "tag": None},
        }


async def test_a_method_without_jsonrpc_answers_invalid_request_with_the_same_id() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {
                "id": "missing-jsonrpc",
                "method": "actions/invoke",
                "params": {
                    "name": "addTodo",
                    "input": {"text": "buy milk"},
                    "invocationId": "missing-jsonrpc",
                },
            }
        )

        answer = await gateway.receive()
        assert answer["id"] == "missing-jsonrpc"
        assert members(answer, "error")["code"] == -32600


async def test_input_that_fails_the_model_is_refused_before_the_handler_runs() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.invoke("addTodo", request_id="inv-1", input_value={"text": ""})

        answer = await gateway.receive()
        assert members(answer, "error")["code"] == -32004


async def test_an_unknown_action_answers_action_not_found() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.invoke("nope", request_id="inv-1", input_value={})

        answer = await gateway.receive()
        assert members(answer, "error")["code"] == -32003


async def test_an_unknown_method_answers_method_not_found() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send({"jsonrpc": "2.0", "id": "x-1", "method": "actions/invented"})

        answer = await gateway.receive()
        assert members(answer, "error")["code"] == -32601


async def test_a_cancelled_invocation_answers_minus_32001() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.invoke("wait", request_id="inv-1", input_value={})
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "method": "actions/cancel",
                "params": {"invocationId": "inv-1"},
            }
        )

        answer = await gateway.receive()
        assert members(answer, "error")["code"] == -32001


async def test_a_read_answers_with_the_resource_value() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {"jsonrpc": "2.0", "id": "r-1", "method": "resources/read", "params": {"name": "cart"}}
        )

        answer = await gateway.receive()
        assert members(answer, "result") == {"value": {"total": 0}}


async def test_reading_a_resource_nobody_declared_answers_not_found() -> None:
    async with listening(todo_application()) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {"jsonrpc": "2.0", "id": "r-1", "method": "resources/read", "params": {"name": "nope"}}
        )

        answer = await gateway.receive()
        error = members(answer, "error")
        assert error["code"] == -32003
        assert error["message"] == "Resource not readable: nope"


async def test_a_subscription_acknowledges_with_null_then_pushes_what_is_published() -> None:
    app = application()

    async def read_cart() -> JsonValue:
        return {"total": 0}

    cart = app.resource("cart", read=read_cart, subscribable=True)

    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "id": "s-1",
                "method": "resources/subscribe",
                "params": {"name": "cart", "subscriptionId": "sub-1"},
            }
        )

        acknowledgement = await gateway.receive()
        assert acknowledgement["id"] == "s-1"
        assert acknowledgement["result"] is None

        await cart.publish({"total": 42})
        update = await gateway.receive()
        assert update["method"] == "resources/updated"
        assert members(update, "params") == {"subscriptionId": "sub-1", "value": {"total": 42}}

        await gateway.send(
            {
                "jsonrpc": "2.0",
                "id": "s-2",
                "method": "resources/unsubscribe",
                "params": {"subscriptionId": "sub-1"},
            }
        )
        assert (await gateway.receive())["result"] is None

        await cart.publish({"total": 99})
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await gateway.receive()


async def test_subscribing_to_a_resource_that_is_not_subscribable_is_refused() -> None:
    app = application()

    async def read_cart() -> JsonValue:
        return {"total": 0}

    app.resource("cart", read=read_cart)

    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "id": "s-1",
                "method": "resources/subscribe",
                "params": {"name": "cart", "subscriptionId": "sub-1"},
            }
        )

        answer = await gateway.receive()
        error = members(answer, "error")
        assert error["code"] == -32003
        assert error["message"] == "Resource not subscribable: cart"


async def test_a_claim_gives_the_session_the_agent_identity() -> None:
    app = application()
    seen: list[str] = []

    @app.action("who", description="Report the caller")
    async def who(raw_input: JsonValue, context: ActionContext) -> JsonValue:
        seen.append(context.agent.id)
        return None

    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "method": "tesseron/claimed",
                "params": {
                    "agent": {"id": "agent_claimed", "name": "claude"},
                    "agentCapabilities": ALL_CAPABILITIES,
                },
            }
        )
        await gateway.invoke("who", request_id="inv-1", input_value={})
        await gateway.receive()

    assert seen == ["agent_claimed"]
    welcome = host.welcome
    assert welcome is not None
    assert welcome.agent.id == "agent_claimed"
    assert welcome.claim_code is None


async def test_a_second_dial_resumes_with_the_credentials_the_welcome_rotated() -> None:
    async with listening(todo_application()) as host:
        async with dial(host) as gateway:
            hello = await gateway.accept_handshake()
            assert hello["method"] == "tesseron/hello"

        async with dial(host) as gateway:
            resume = await gateway.receive()
            assert resume["method"] == "tesseron/resume"
            params = members(resume, "params")
            assert params["sessionId"] == SESSION_ID
            assert params["resumeToken"] == RESUME_TOKEN
            # The manifest repeats, because a restarted application may have changed it.
            assert len(entries(params, "actions")) == 2


async def test_a_refused_resume_falls_back_to_a_fresh_hello() -> None:
    async with listening(todo_application()) as host:
        async with dial(host) as gateway:
            await gateway.accept_handshake()

        async with dial(host) as gateway:
            resume = await gateway.receive()
            await gateway.refuse(resume, -32011, "Resume failed")

            hello = await gateway.receive()
            assert hello["method"] == "tesseron/hello"
            await gateway.answer(
                hello,
                {
                    "sessionId": "s_test_0002",
                    "protocolVersion": "1.2.0",
                    "capabilities": ALL_CAPABILITIES,
                    "agent": {"id": "agent_test", "name": "test-runner"},
                    "claimCode": CLAIM_CODE,
                    "resumeToken": "rt_test_0002",
                },
            )

            await gateway.invoke("addTodo", request_id="inv-1", input_value={"text": "again"})
            await gateway.receive()
            answer = await gateway.receive()
            assert members(answer, "result")["invocationId"] == "inv-1"


async def test_a_protocol_mismatch_ends_the_connection_rather_than_retrying() -> None:
    async with listening(todo_application()) as host:
        async with dial(host) as gateway:
            await gateway.accept_handshake()

        async with dial(host) as gateway:
            resume = await gateway.receive()
            await gateway.refuse(resume, -32000, "Protocol mismatch")

            # A refusal is about this application, not this socket, so a fresh hello here
            # would only loop. The host closes and waits for the next dial.
            with pytest.raises(ConnectionClosed):
                await gateway.receive()


async def test_an_upgrade_without_the_gateway_subprotocol_is_refused() -> None:
    async with listening(todo_application()) as host:
        with pytest.raises(InvalidStatus) as refusal:
            async with connect(host.url):
                pass
        assert refusal.value.response.status_code == 400


async def test_an_unusable_application_id_never_binds() -> None:
    app = TesseronApp(id="Todo", name="Todo", manifest=ManifestPublication.disabled())

    with pytest.raises(InvalidApplicationIdError):
        await app.listen()


async def test_one_name_cannot_be_registered_twice() -> None:
    app = application()

    async def read_cart() -> JsonValue:
        return {}

    @app.action("addTodo")
    async def first(raw_input: JsonValue, context: ActionContext) -> JsonValue:
        return None

    with pytest.raises(DuplicateNameError):

        @app.action("addTodo")
        async def second(raw_input: JsonValue, context: ActionContext) -> JsonValue:
            return None

    app.resource("cart", read=read_cart)
    with pytest.raises(DuplicateNameError):
        app.resource("cart", read=read_cart)


def test_a_handler_that_does_not_take_input_and_context_is_refused() -> None:
    app = application()

    async def only_input(raw_input: JsonValue) -> JsonValue:
        return None

    with pytest.raises(HostError, match="input, context"):
        app.action("addTodo")(only_input)


def welcome_event(app: TesseronApp) -> asyncio.Event:
    welcomed = asyncio.Event()
    app.add_event_listener(
        lambda event: welcomed.set() if isinstance(event, WelcomeEvent) else None
    )
    return welcomed


async def test_registering_an_action_after_welcome_pushes_the_full_action_manifest() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    async with listening(app) as host, dial(host) as gateway:
        hello = await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)

        @host.action(
            "echo", description="Echo input", output_schema={"type": "object"}, timeout_ms=500
        )
        async def echo(parsed: AddTodo, context: ActionContext) -> JsonValue:
            return {"text": parsed.text}

        notification = await gateway.receive()
        expected_notification: JsonObject = {
            "jsonrpc": "2.0",
            "method": "actions/list_changed",
            "params": {
                "actions": [
                    *entries(members(hello, "params"), "actions"),
                    {
                        "name": "echo",
                        "description": "Echo input",
                        "inputSchema": AddTodo.model_json_schema(mode="validation"),
                        "outputSchema": {"type": "object"},
                        "timeoutMs": 500,
                    },
                ]
            },
        }
        assert notification == expected_notification
        await gateway.invoke("echo", request_id="echo-1", input_value={"text": "new"})
        assert members(await gateway.receive(), "result")["output"] == {"text": "new"}
        await gateway.invoke("echo", request_id="echo-2", input_value={"text": ""})
        assert members(await gateway.receive(), "error")["code"] == -32004
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)


async def test_removing_a_resource_after_welcome_notifies_and_unknown_names_are_silent() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)
        assert host.remove_resource("cart") is True
        assert await gateway.receive() == {
            "jsonrpc": "2.0",
            "method": "resources/list_changed",
            "params": {"resources": []},
        }
        assert host.remove_resource("unknown") is False
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)


async def test_registrations_before_welcome_are_carried_by_hello_and_send_nothing() -> None:
    app = application()
    welcomed = welcome_event(app)
    async with listening(app) as host:

        @host.action("echo")
        async def echo(raw_input: JsonValue, context: ActionContext) -> JsonValue:
            return raw_input

        async def read_cart() -> JsonValue:
            return {}

        host.resource("cart", read=read_cart)
        async with dial(host) as gateway:
            hello = await gateway.accept_handshake()
            assert hello["method"] == "tesseron/hello"
            assert entries(members(hello, "params"), "actions") == [
                {"name": "echo", "description": "", "inputSchema": {}}
            ]
            assert entries(members(hello, "params"), "resources") == [
                {"name": "cart", "description": "", "subscribable": False}
            ]
            await asyncio.wait_for(welcomed.wait(), 1)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(gateway.receive(), 0.1)


async def test_replacing_an_action_swaps_its_handler_and_keeps_its_position() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    async with listening(app) as host, dial(host) as gateway:
        hello = await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)

        @host.action("addTodo", description="First replacement")
        async def first(raw_input: JsonValue, context: ActionContext) -> JsonValue:
            return "first"

        first_manifest = entries(members(await gateway.receive(), "params"), "actions")
        assert first_manifest[0] == {
            "name": "addTodo",
            "description": "First replacement",
            "inputSchema": {},
        }

        @host.action("addTodo", description="Second replacement")
        async def second(raw_input: JsonValue, context: ActionContext) -> JsonValue:
            return "second"

        notification = await gateway.receive()
        assert notification["method"] == "actions/list_changed"
        assert entries(members(notification, "params"), "actions") == [
            {"name": "addTodo", "description": "Second replacement", "inputSchema": {}},
            entries(members(hello, "params"), "actions")[1],
        ]
        await gateway.invoke("addTodo", request_id="replacement", input_value={})
        assert members(await gateway.receive(), "result")["output"] == "second"
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)


@pytest.mark.parametrize("remove", [False, True])
async def test_replacing_a_subscribed_resource_stops_its_subscriptions(remove: bool) -> None:
    app = application()
    welcomed = welcome_event(app)
    stopped: list[str] = []

    async def read_cart() -> JsonValue:
        return {}

    def subscribe(emit: Emit) -> Unsubscribe:
        return lambda: stopped.append("cart")

    cart = app.resource("cart", read=read_cart, subscribe=subscribe)
    other = app.resource("other", read=read_cart, subscribable=True)
    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)
        for subscription_id, name in [("cart-1", "cart"), ("other-1", "other"), ("cart-2", "cart")]:
            await gateway.send(
                {
                    "jsonrpc": "2.0",
                    "id": subscription_id,
                    "method": "resources/subscribe",
                    "params": {"name": name, "subscriptionId": subscription_id},
                }
            )
            assert (await gateway.receive())["result"] is None
        await cart.publish("before")
        assert members(await gateway.receive(), "params")["subscriptionId"] == "cart-1"
        assert members(await gateway.receive(), "params")["subscriptionId"] == "cart-2"

        if remove:
            assert host.remove_resource("cart") is True
        else:
            replacement = host.resource(
                "cart", read=read_cart, description="New", subscribable=True
            )
            await replacement.publish("not subscribed yet")
        notification = await gateway.receive()
        assert notification["method"] == "resources/list_changed"
        resources: list[JsonValue] = [{"name": "other", "description": "", "subscribable": True}]
        if not remove:
            resources.insert(0, {"name": "cart", "description": "New", "subscribable": True})
        assert members(notification, "params") == {"resources": resources}
        assert stopped == ["cart", "cart"]
        await cart.publish("after")
        await other.publish("still subscribed")
        assert members(await gateway.receive(), "params") == {
            "subscriptionId": "other-1",
            "value": "still subscribed",
        }
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)
    assert stopped == ["cart", "cart"]


async def test_removing_a_resource_continues_after_a_subscription_cleanup_raises() -> None:
    app = application()
    welcomed = welcome_event(app)
    cleanup_calls: list[int] = []
    subscription_count = 0

    async def read_cart() -> JsonValue:
        return {}

    def subscribe(emit: Emit) -> Unsubscribe:
        nonlocal subscription_count
        subscription_count += 1
        cleanup_number = subscription_count

        def cleanup() -> None:
            cleanup_calls.append(cleanup_number)
            if cleanup_number == 1:
                raise RuntimeError("cleanup failed")

        return cleanup

    cart = app.resource("cart", read=read_cart, subscribe=subscribe)
    async with listening(app) as host, dial(host) as gateway:
        await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)
        for subscription_id in ["cart-1", "cart-2"]:
            await gateway.send(
                {
                    "jsonrpc": "2.0",
                    "id": subscription_id,
                    "method": "resources/subscribe",
                    "params": {"name": "cart", "subscriptionId": subscription_id},
                }
            )
            assert (await gateway.receive())["result"] is None

        assert host.remove_resource("cart") is True
        assert await gateway.receive() == {
            "jsonrpc": "2.0",
            "method": "resources/list_changed",
            "params": {"resources": []},
        }
        assert cleanup_calls == [1, 2]
        await cart.publish("after")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)


async def test_disconnect_continues_after_a_subscription_cleanup_raises() -> None:
    app = application()
    welcomed = welcome_event(app)
    disconnected = asyncio.Event()
    app.add_event_listener(
        lambda event: disconnected.set() if isinstance(event, DisconnectedEvent) else None
    )

    async def read_cart() -> JsonValue:
        return {}

    def subscribe(emit: Emit) -> Unsubscribe:
        def cleanup() -> None:
            raise RuntimeError("cleanup failed")

        return cleanup

    app.resource("cart", read=read_cart, subscribe=subscribe)
    async with listening(app) as host:
        async with dial(host) as gateway:
            await gateway.accept_handshake()
            await asyncio.wait_for(welcomed.wait(), 1)
            await gateway.send(
                {
                    "jsonrpc": "2.0",
                    "id": "cart-1",
                    "method": "resources/subscribe",
                    "params": {"name": "cart", "subscriptionId": "cart-1"},
                }
            )
            assert (await gateway.receive())["result"] is None

        await asyncio.wait_for(disconnected.wait(), 1)
        assert host._shared._session is None

        async def read_stock() -> JsonValue:
            return {}

        host.resource("stock", read=read_stock)
        host._shared.forget_resume_credentials()
        async with dial(host) as gateway:
            hello = await gateway.receive()
            assert hello["method"] == "tesseron/hello"
            assert entries(members(hello, "params"), "resources") == [
                {"name": "cart", "description": "", "subscribable": True},
                {"name": "stock", "description": "", "subscribable": False},
            ]


async def test_removing_an_action_notifies_once_and_stops_new_invocations() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    async with listening(app) as host, dial(host) as gateway:
        hello = await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)
        assert host.remove_action("addTodo") is True
        assert await gateway.receive() == {
            "jsonrpc": "2.0",
            "method": "actions/list_changed",
            "params": {"actions": entries(members(hello, "params"), "actions")[1:]},
        }
        assert host.remove_action("unknown") is False
        await gateway.invoke("addTodo", request_id="removed", input_value={})
        assert members(await gateway.receive(), "error")["code"] == -32003
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)


@pytest.mark.parametrize("refuse_resume", [False, True])
async def test_disconnected_mutations_resume_silently_until_an_accepted_welcome(
    refuse_resume: bool,
) -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    disconnected = asyncio.Event()
    app.add_event_listener(
        lambda event: disconnected.set() if isinstance(event, DisconnectedEvent) else None
    )
    async with listening(app) as host:
        async with dial(host) as gateway:
            await gateway.accept_handshake()
            await asyncio.wait_for(welcomed.wait(), 1)
        await asyncio.wait_for(disconnected.wait(), 1)
        welcomed.clear()
        assert host.remove_action("wait") is True
        async with dial(host) as gateway:
            resume = await gateway.receive()
            assert resume["method"] == "tesseron/resume"
            actions = entries(members(resume, "params"), "actions")
            assert len(actions) == 1
            assert isinstance(actions[0], dict) and actions[0]["name"] == "addTodo"
            assert host.remove_resource("cart") is True
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(gateway.receive(), 0.1)
            handshake = resume
            if refuse_resume:
                await gateway.refuse(resume, -32011, "Resume failed")
                handshake = await gateway.receive()
                assert handshake["method"] == "tesseron/hello"
                assert members(handshake, "params")["resources"] == []
            await gateway.answer(
                handshake,
                {
                    "sessionId": SESSION_ID,
                    "protocolVersion": "1.2.0",
                    "capabilities": ALL_CAPABILITIES,
                    "agent": {"id": "agent_test", "name": "test-runner"},
                    "resumeToken": "next-token",
                },
            )
            await asyncio.wait_for(welcomed.wait(), 1)
            assert host.remove_action("addTodo") is True
            assert await gateway.receive() == {
                "jsonrpc": "2.0",
                "method": "actions/list_changed",
                "params": {"actions": []},
            }
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(gateway.receive(), 0.1)


async def test_a_rejected_resume_never_attaches_a_session_for_mutations() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    refused = asyncio.Event()
    async with listening(app) as host:

        def on_event(event: HostEvent) -> None:
            if isinstance(event, HandshakeFailedEvent):
                assert host.remove_action("addTodo") is True
                refused.set()

        app.add_event_listener(on_event)
        async with dial(host) as gateway:
            await gateway.accept_handshake()
            await asyncio.wait_for(welcomed.wait(), 1)
        async with dial(host) as gateway:
            resume = await gateway.receive()
            assert resume["method"] == "tesseron/resume"
            await gateway.refuse(resume, -32000, "Protocol mismatch")
            await asyncio.wait_for(refused.wait(), 1)
            with pytest.raises(ConnectionClosed):
                await gateway.receive()


async def test_registering_a_resource_after_welcome_appends_the_full_manifest() -> None:
    app = todo_application()
    welcomed = welcome_event(app)
    async with listening(app) as host, dial(host) as gateway:
        hello = await gateway.accept_handshake()
        await asyncio.wait_for(welcomed.wait(), 1)

        async def read_stock() -> JsonValue:
            return {"remaining": 3}

        host.resource("stock", read=read_stock, description="Stock")
        expected_notification: JsonObject = {
            "jsonrpc": "2.0",
            "method": "resources/list_changed",
            "params": {
                "resources": [
                    *entries(members(hello, "params"), "resources"),
                    {"name": "stock", "description": "Stock", "subscribable": False},
                ]
            },
        }
        assert await gateway.receive() == expected_notification
        await gateway.send(
            {
                "jsonrpc": "2.0",
                "id": "read-stock",
                "method": "resources/read",
                "params": {"name": "stock"},
            }
        )
        assert members(await gateway.receive(), "result")["value"] == {"remaining": 3}
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gateway.receive(), 0.1)
