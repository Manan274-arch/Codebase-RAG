"""Generate the frozen synthetic Retrieval Benchmark v2 fixture."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT / "tests" / "fixtures" / "retrieval_eval_repo_v2"
BENCHMARK = ROOT / "tests" / "fixtures" / "retrieval_eval_v2.json"


def backend_route(module: str, body: str) -> str:
    return f"from fastapi import APIRouter\nfrom backend.app.services import {module}\n\nrouter = APIRouter()\n\n{body}\n"


FILES: dict[str, str] = {
    "README.md": "# Northstar Commerce\nSynthetic order-management platform.\n",
    ".env.example": "PAYMENT_PROVIDER_URL=https://payments.invalid\nJWT_SECRET=local-only\n",
    "frontend/package.json": '{"name":"northstar-web","private":true}\n',
    "backend/app/routes/auth.py": backend_route("auth_service", '''@router.post("/api/auth/login")
def login(credentials: dict):
    return auth_service.authenticate_user(credentials)

@router.post("/api/auth/refresh")
def refresh_session(payload: dict):
    return auth_service.rotate_session(payload["refresh_token"])'''),
    "backend/app/routes/profile.py": backend_route("profile_service", '''@router.get("/api/profile")
def current_profile(user_id: str):
    return profile_service.load_profile(user_id)

@router.patch("/api/profile")
def update_profile(user_id: str, changes: dict):
    return profile_service.update_profile(user_id, changes)'''),
    "backend/app/routes/orders.py": backend_route("order_service", '''@router.post("/api/orders")
def create_order(payload: dict):
    return order_service.place_order(payload)

@router.get("/api/orders/{order_id}")
def get_order(order_id: str):
    return order_service.load_order(order_id)

@router.delete("/api/orders/{order_id}")
def cancel_order(order_id: str):
    return order_service.cancel_if_allowed(order_id)'''),
    "backend/app/routes/checkout.py": backend_route("checkout_service", '''@router.post("/api/checkout")
def submit_checkout(payload: dict):
    return checkout_service.complete_checkout(payload)'''),
    "backend/app/routes/payments.py": backend_route("payment_gateway", '''@router.post("/api/payments/authorize")
def authorize_payment(payload: dict):
    return payment_gateway.process_charge(payload)

@router.post("/api/payments/webhook")
def payment_webhook(event: dict):
    return payment_gateway.accept_provider_event(event)'''),
    "backend/app/routes/inventory.py": backend_route("inventory_service", '''@router.get("/api/inventory/{sku}")
def inventory_status(sku: str):
    return inventory_service.available_units(sku)

@router.patch("/api/inventory/{sku}/reserve")
def reserve_inventory(sku: str, payload: dict):
    return inventory_service.reserve_stock(sku, payload["quantity"])'''),
    "backend/app/routes/shipping.py": backend_route("shipping_service", '''@router.get("/api/shipping/{order_id}")
def shipment_status(order_id: str):
    return shipping_service.track_order(order_id)'''),
    "backend/app/routes/refunds.py": backend_route("refund_service", '''@router.post("/api/refunds")
def request_refund(payload: dict):
    return refund_service.open_refund(payload)'''),
    "backend/app/routes/admin.py": backend_route("refund_service", '''@router.post("/api/admin/refunds/{refund_id}/approve")
def approve_refund(refund_id: str, admin_id: str):
    return refund_service.approve_refund(refund_id, admin_id)'''),
    "backend/app/routes/health.py": '''from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok", "database": "reachable"}
''',
    "backend/app/services/auth_service.py": '''from backend.app.repositories import user_repository
from backend.app.utils import jwt_tokens

def authenticate_user(credentials: dict) -> dict:
    user = user_repository.find_by_email(credentials["email"])
    if not user or not user_repository.verify_password(user, credentials["password"]):
        raise ValueError("invalid credentials")
    return jwt_tokens.issue_token_pair(user["id"])

def rotate_session(refresh_token: str) -> dict:
    claims = jwt_tokens.decode_refresh_token(refresh_token)
    return jwt_tokens.issue_token_pair(claims["sub"])
''',
    "backend/app/services/profile_service.py": '''from backend.app.repositories import user_repository

def load_profile(user_id: str) -> dict:
    return user_repository.get_public_profile(user_id)

def update_profile(user_id: str, changes: dict) -> dict:
    allowed = {key: value for key, value in changes.items() if key in {"name", "timezone"}}
    return user_repository.update_profile(user_id, allowed)
''',
    "backend/app/services/order_service.py": '''from backend.app.repositories import order_repository
from backend.app.services import inventory_service, notification_service

def place_order(payload: dict) -> dict:
    inventory_service.reserve_items(payload["items"])
    order = order_repository.insert_order(payload)
    notification_service.queue_order_confirmation(order)
    return order

def load_order(order_id: str) -> dict:
    return order_repository.get_order(order_id)

def cancel_if_allowed(order_id: str) -> dict:
    order = order_repository.get_order(order_id)
    if order["status"] not in {"pending", "confirmed"}:
        raise ValueError("order can no longer be cancelled")
    inventory_service.release_items(order["items"])
    return order_repository.set_status(order_id, "cancelled")
''',
    "backend/app/services/checkout_service.py": '''from backend.app.services import discount_service, order_service, payment_gateway

def complete_checkout(payload: dict) -> dict:
    total = discount_service.final_total(payload["items"], payload.get("coupon"))
    authorization = payment_gateway.process_charge({"amount": total, "token": payload["payment_token"]})
    return order_service.place_order({**payload, "total": total, "authorization": authorization["id"]})
''',
    "backend/app/services/payment_gateway.py": '''from backend.app.repositories import payment_repository

TRANSIENT_CODES = {"timeout", "rate_limited"}

def process_charge(request: dict) -> dict:
    response = _provider_authorize(request)
    if response["status"] in TRANSIENT_CODES:
        payment_repository.mark_for_retry(request)
        raise RuntimeError("payment temporarily unavailable")
    if response["status"] != "approved":
        raise ValueError("payment authorization failed")
    return payment_repository.record_authorization(response)

def accept_provider_event(event: dict) -> dict:
    if payment_repository.webhook_seen(event["id"]):
        return {"duplicate": True}
    payment_repository.record_webhook(event)
    return {"accepted": True}

def _provider_authorize(request: dict) -> dict:
    return {"id": "auth_generated", "status": "approved", **request}
''',
    "backend/app/services/inventory_service.py": '''from backend.app.repositories import inventory_repository

def available_units(sku: str) -> int:
    return inventory_repository.quantity_for(sku)

def reserve_stock(sku: str, quantity: int) -> dict:
    return inventory_repository.decrement_available(sku, quantity)

def reserve_items(items: list[dict]) -> None:
    for item in items:
        reserve_stock(item["sku"], item["quantity"])

def release_items(items: list[dict]) -> None:
    for item in items:
        inventory_repository.increment_available(item["sku"], item["quantity"])
''',
    "backend/app/services/shipping_service.py": '''from backend.app.repositories import shipment_repository

def track_order(order_id: str) -> dict:
    return shipment_repository.find_by_order(order_id)

def schedule_dispatch(order_id: str, address: dict) -> dict:
    return shipment_repository.create_shipment(order_id, address)
''',
    "backend/app/services/refund_service.py": '''from backend.app.repositories import refund_repository, payment_repository

def open_refund(payload: dict) -> dict:
    return refund_repository.insert_request(payload)

def approve_refund(refund_id: str, admin_id: str) -> dict:
    refund = refund_repository.mark_approved(refund_id, admin_id)
    payment_repository.return_funds(refund["payment_id"], refund["amount"])
    return refund
''',
    "backend/app/services/notification_service.py": '''def queue_order_confirmation(order: dict) -> dict:
    return {"job": "order_confirmation", "order_id": order["id"]}

def queue_shipping_update(shipment: dict) -> dict:
    return {"job": "shipping_update", "shipment_id": shipment["id"]}
''',
    "backend/app/services/discount_service.py": '''from decimal import Decimal

def final_total(items: list[dict], coupon: str | None) -> Decimal:
    subtotal = sum(Decimal(str(item["price"])) * item["quantity"] for item in items)
    discount = Decimal("0.10") if coupon == "SAVE10" else Decimal("0")
    return subtotal * (Decimal("1") - discount)

def preview_total(items: list[dict]) -> Decimal:
    return final_total(items, None)
''',
    "backend/app/repositories/user_repository.py": '''USERS: dict[str, dict] = {}

def find_by_email(email: str) -> dict | None:
    return next((user for user in USERS.values() if user["email"] == email), None)

def verify_password(user: dict, password: str) -> bool:
    return user.get("password_hash") == f"hashed:{password}"

def get_public_profile(user_id: str) -> dict:
    user = USERS[user_id]
    return {"id": user_id, "name": user["name"], "timezone": user["timezone"]}

def update_profile(user_id: str, changes: dict) -> dict:
    USERS[user_id].update(changes)
    return get_public_profile(user_id)
''',
    "backend/app/repositories/order_repository.py": '''ORDERS: dict[str, dict] = {}

def insert_order(payload: dict) -> dict:
    order_id = f"ord_{len(ORDERS) + 1}"
    ORDERS[order_id] = {"id": order_id, "status": "confirmed", **payload}
    return ORDERS[order_id]

def get_order(order_id: str) -> dict:
    return ORDERS[order_id]

def set_status(order_id: str, status: str) -> dict:
    ORDERS[order_id]["status"] = status
    return ORDERS[order_id]
''',
    "backend/app/repositories/inventory_repository.py": '''STOCK: dict[str, int] = {}

def quantity_for(sku: str) -> int:
    return STOCK.get(sku, 0)

def decrement_available(sku: str, quantity: int) -> dict:
    if quantity_for(sku) < quantity:
        raise ValueError("insufficient inventory")
    STOCK[sku] -= quantity
    return {"sku": sku, "available": STOCK[sku]}

def increment_available(sku: str, quantity: int) -> dict:
    STOCK[sku] = quantity_for(sku) + quantity
    return {"sku": sku, "available": STOCK[sku]}
''',
    "backend/app/repositories/payment_repository.py": '''AUTHORIZATIONS: dict[str, dict] = {}
WEBHOOKS: set[str] = set()

def record_authorization(response: dict) -> dict:
    AUTHORIZATIONS[response["id"]] = response
    return response

def mark_for_retry(request: dict) -> None:
    AUTHORIZATIONS[request["token"]] = {**request, "status": "retry"}

def webhook_seen(event_id: str) -> bool:
    return event_id in WEBHOOKS

def record_webhook(event: dict) -> None:
    WEBHOOKS.add(event["id"])

def return_funds(payment_id: str, amount: float) -> dict:
    return {"payment_id": payment_id, "refunded": amount}
''',
    "backend/app/repositories/shipment_repository.py": '''SHIPMENTS: dict[str, dict] = {}

def create_shipment(order_id: str, address: dict) -> dict:
    shipment = {"id": f"ship_{len(SHIPMENTS)+1}", "order_id": order_id, "address": address, "status": "queued"}
    SHIPMENTS[shipment["id"]] = shipment
    return shipment

def find_by_order(order_id: str) -> dict:
    return next(item for item in SHIPMENTS.values() if item["order_id"] == order_id)
''',
    "backend/app/repositories/refund_repository.py": '''REFUNDS: dict[str, dict] = {}

def insert_request(payload: dict) -> dict:
    refund_id = f"ref_{len(REFUNDS)+1}"
    REFUNDS[refund_id] = {"id": refund_id, "status": "requested", **payload}
    return REFUNDS[refund_id]

def mark_approved(refund_id: str, admin_id: str) -> dict:
    REFUNDS[refund_id].update(status="approved", approved_by=admin_id)
    return REFUNDS[refund_id]
''',
    "backend/app/middleware/authentication.py": '''from backend.app.utils.jwt_tokens import decode_access_token

def require_authenticated_user(authorization: str) -> str:
    if not authorization.startswith("Bearer "):
        raise PermissionError("missing bearer token")
    claims = decode_access_token(authorization.removeprefix("Bearer "))
    if claims.get("expired"):
        raise PermissionError("expired session")
    return claims["sub"]
''',
    "backend/app/middleware/rate_limit.py": '''COUNTERS: dict[str, int] = {}

def enforce_request_limit(client_id: str, limit: int = 100) -> None:
    COUNTERS[client_id] = COUNTERS.get(client_id, 0) + 1
    if COUNTERS[client_id] > limit:
        raise RuntimeError("request rate exceeded")
''',
    "backend/app/utils/jwt_tokens.py": '''from datetime import datetime, timedelta

def issue_token_pair(user_id: str) -> dict:
    return {"access_token": f"access:{user_id}", "refresh_token": f"refresh:{user_id}"}

def decode_access_token(token: str) -> dict:
    return {"sub": token.split(":")[-1], "expired": False}

def decode_refresh_token(token: str) -> dict:
    if not token.startswith("refresh:"):
        raise ValueError("invalid refresh token")
    return {"sub": token.split(":")[-1], "expires_at": datetime.now() + timedelta(days=7)}
''',
    "backend/app/utils/money.py": '''from decimal import Decimal, ROUND_HALF_UP

def round_currency(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def format_currency(amount: Decimal, currency: str = "USD") -> str:
    return f"{currency} {round_currency(amount)}"
''',
    "backend/app/utils/idempotency.py": '''PROCESSED_KEYS: set[str] = set()

def claim_once(key: str) -> bool:
    if key in PROCESSED_KEYS:
        return False
    PROCESSED_KEYS.add(key)
    return True
''',
    "frontend/src/api/auth.ts": '''import axios from "axios";
export async function signIn(email: string, password: string) { return axios.post("/api/auth/login", { email, password }); }
export async function renewSession(refreshToken: string) { return axios.post("/api/auth/refresh", { refresh_token: refreshToken }); }
''',
    "frontend/src/api/profile.ts": '''import axios from "axios";
export async function loadCurrentProfile() { return axios.get("/api/profile"); }
export async function saveProfile(changes: object) { return axios.patch("/api/profile", changes); }
''',
    "frontend/src/api/orders.ts": '''import axios from "axios";
export async function submitOrder(payload: object) { return axios.post("/api/orders", payload); }
export async function fetchOrder(orderId: string) { return axios.get(`/api/orders/${orderId}`); }
export async function requestCancellation(orderId: string) { return axios.delete(`/api/orders/${orderId}`); }
''',
    "frontend/src/api/checkout.ts": '''import axios from "axios";
export async function sendCheckout(cart: object) { return axios.post("/api/checkout", cart); }
''',
    "frontend/src/api/payments.ts": '''import axios from "axios";
export async function authorizeCard(payload: object) { return axios.post("/api/payments/authorize", payload); }
export async function previewPayment(payload: object) { return { status: "preview", payload }; }
''',
    "frontend/src/api/inventory.ts": '''import axios from "axios";
export async function loadAvailability(sku: string) { return axios.get(`/api/inventory/${sku}`); }
export async function holdInventory(sku: string, quantity: number) { return axios.patch(`/api/inventory/${sku}/reserve`, { quantity }); }
''',
    "frontend/src/api/shipping.ts": '''import axios from "axios";
export async function trackShipment(orderId: string) { return axios.get(`/api/shipping/${orderId}`); }
''',
    "frontend/src/api/refunds.ts": '''import axios from "axios";
export async function submitRefund(payload: object) { return axios.post("/api/refunds", payload); }
''',
    "frontend/src/api/admin.ts": '''import axios from "axios";
export async function approveRefund(refundId: string) { return axios.post(`/api/admin/refunds/${refundId}/approve`, {}); }
''',
    "frontend/src/pages/CheckoutPage.ts": '''import { sendCheckout } from "../api/checkout";
export async function completePurchase(cart: object) { const result = await sendCheckout(cart); return result.data; }
''',
    "frontend/src/pages/OrderHistory.ts": '''import { fetchOrder } from "../api/orders";
export async function openOrder(orderId: string) { return (await fetchOrder(orderId)).data; }
''',
    "frontend/src/pages/AdminRefund.ts": '''import { approveRefund } from "../api/admin";
export async function confirmRefund(refundId: string) { return approveRefund(refundId); }
''',
    "frontend/src/components/LoginForm.ts": '''import { signIn } from "../api/auth";
export async function submitCredentials(email: string, password: string) { return signIn(email, password); }
''',
    "frontend/src/hooks/useSession.ts": '''import { renewSession } from "../api/auth";
export async function restoreSession(refreshToken: string) { return renewSession(refreshToken); }
''',
    "frontend/src/hooks/useInventory.ts": '''import { loadAvailability } from "../api/inventory";
export async function refreshStock(sku: string) { return loadAvailability(sku); }
''',
    "frontend/src/utils/currency.ts": '''export function displayMoney(amount: number, currency = "USD") { return `${currency} ${amount.toFixed(2)}`; }
''',
    "frontend/src/utils/orderPreview.ts": '''export function buildOrderPreview(items: object[]) { return { items, persisted: false, status: "preview" }; }
''',
    "worker/jobs/payment_retry.py": '''from backend.app.services.payment_gateway import process_charge

def retry_deferred_charge(payload: dict, attempt: int) -> dict:
    if attempt > 3:
        raise RuntimeError("payment retry exhausted")
    return process_charge(payload)
''',
    "worker/jobs/shipment_dispatch.py": '''from backend.app.services.shipping_service import schedule_dispatch

def dispatch_confirmed_order(order: dict) -> dict:
    return schedule_dispatch(order["id"], order["shipping_address"])
''',
    "worker/jobs/notification_delivery.py": '''def deliver_notification(job: dict) -> dict:
    return {"delivered": True, "template": job["job"], "recipient": job.get("user_id")}
''',
    "worker/jobs/inventory_reconcile.py": '''from backend.app.repositories.inventory_repository import quantity_for

def reconcile_catalog_stock(skus: list[str]) -> dict[str, int]:
    return {sku: quantity_for(sku) for sku in skus}
''',
    "backend/tests/payment_mock.py": '''def fake_provider_authorize(payload: dict) -> dict:
    return {"id": "mock", "status": "approved", **payload}
''',
    "backend/tests/order_factory.py": '''def build_order(overrides: dict | None = None) -> dict:
    return {"id": "test-order", "status": "pending", **(overrides or {})}
''',
    "backend/tests/auth_stub.py": '''def authenticated_user() -> dict:
    return {"id": "test-user", "email": "developer@example.test"}
''',
    "backend/tests/inventory_fake.py": '''class FakeInventoryRepository:
    def decrement_available(self, sku: str, quantity: int) -> dict:
        return {"sku": sku, "available": 999 - quantity}
''',
}


def cid(path: str) -> str:
    return f"{path}::0"


def q(text: str, primary: list[str], supporting: list[str] | None = None) -> tuple[str, dict[str, int]]:
    relevance = {cid(path): 2 for path in primary}
    relevance.update({cid(path): 1 for path in supporting or []})
    return text, relevance


QUERIES: dict[str, list[tuple[str, dict[str, int]]]] = {
    "lexical": [
        q("Where is `rotate_session` defined?", ["backend/app/services/auth_service.py"]),
        q("Which file defines `InventoryRepository`-style stock persistence through `decrement_available`?", ["backend/app/repositories/inventory_repository.py"]),
        q("Where is the `/health` route registered?", ["backend/app/routes/health.py"]),
        q("Find the `complete_checkout` implementation.", ["backend/app/services/checkout_service.py"]),
        q("Where is `requestCancellation` defined?", ["frontend/src/api/orders.ts"]),
        q("Which module defines `record_authorization`?", ["backend/app/repositories/payment_repository.py"]),
        q("Find the `approve_refund` service function.", ["backend/app/services/refund_service.py"]),
        q("Where is `require_authenticated_user` implemented?", ["backend/app/middleware/authentication.py"]),
        q("Which worker defines `retry_deferred_charge`?", ["worker/jobs/payment_retry.py"]),
        q("Find the `buildOrderPreview` helper.", ["frontend/src/utils/orderPreview.ts"]),
        q("Where is `queue_order_confirmation` defined?", ["backend/app/services/notification_service.py"]),
        q("Which file contains `decode_refresh_token`?", ["backend/app/utils/jwt_tokens.py"]),
        q("Where is `trackShipment` implemented?", ["frontend/src/api/shipping.ts"]),
        q("Find the `mark_approved` repository function.", ["backend/app/repositories/refund_repository.py"]),
        q("Where is `displayMoney` defined?", ["frontend/src/utils/currency.ts"]),
        q("Which route handler is named `inventory_status`?", ["backend/app/routes/inventory.py"]),
        q("Find `dispatch_confirmed_order`.", ["worker/jobs/shipment_dispatch.py"]),
        q("Where is `claim_once` defined?", ["backend/app/utils/idempotency.py"]),
    ],
    "semantic": [
        q("What code stops an expired login session from reaching protected work?", ["backend/app/middleware/authentication.py"], ["backend/app/utils/jwt_tokens.py"]),
        q("Where are bad email and password combinations rejected?", ["backend/app/services/auth_service.py"], ["backend/app/repositories/user_repository.py"]),
        q("What decides whether a customer can still cancel an order?", ["backend/app/services/order_service.py"]),
        q("Where is a temporary card-provider failure queued for another attempt?", ["backend/app/services/payment_gateway.py", "backend/app/repositories/payment_repository.py"], ["worker/jobs/payment_retry.py"]),
        q("What code prevents the same provider callback from being handled twice?", ["backend/app/services/payment_gateway.py", "backend/app/repositories/payment_repository.py"]),
        q("Where is the amount charged after applying a coupon calculated?", ["backend/app/services/discount_service.py", "backend/app/services/checkout_service.py"]),
        q("What reduces sellable stock when merchandise is held?", ["backend/app/repositories/inventory_repository.py"], ["backend/app/services/inventory_service.py"]),
        q("Where is stock restored after a purchase is cancelled?", ["backend/app/services/inventory_service.py"], ["backend/app/services/order_service.py"]),
        q("What writes a newly submitted purchase into durable application state?", ["backend/app/repositories/order_repository.py"], ["backend/app/services/order_service.py"]),
        q("Where is sensitive profile input restricted to approved fields?", ["backend/app/services/profile_service.py"]),
        q("What initiates delivery preparation after an order is confirmed?", ["worker/jobs/shipment_dispatch.py", "backend/app/services/shipping_service.py"]),
        q("Where does an approved reimbursement return money to the original payment?", ["backend/app/services/refund_service.py", "backend/app/repositories/payment_repository.py"]),
        q("What browser behavior restores a session without asking for a password again?", ["frontend/src/hooks/useSession.ts", "frontend/src/api/auth.ts"]),
        q("Where does checkout combine pricing, payment, and order creation?", ["backend/app/services/checkout_service.py"]),
        q("What code rejects reservation when there are too few units?", ["backend/app/repositories/inventory_repository.py"]),
        q("Where are excessive requests from one client blocked?", ["backend/app/middleware/rate_limit.py"]),
        q("What turns an order confirmation into asynchronous notification work?", ["backend/app/services/notification_service.py"], ["worker/jobs/notification_delivery.py"]),
        q("Where does the UI format monetary values for display?", ["frontend/src/utils/currency.ts"], ["backend/app/utils/money.py"]),
    ],
    "structural": [
        q("Which service owns the logic invoked by the order creation handler?", ["backend/app/services/order_service.py"], ["backend/app/routes/orders.py"]),
        q("Where is the handler for the endpoint returning the signed-in user's profile?", ["backend/app/routes/profile.py"], ["backend/app/services/profile_service.py"]),
        q("Which repository method actually persists inventory decrements?", ["backend/app/repositories/inventory_repository.py"], ["backend/app/services/inventory_service.py"]),
        q("What service is imported by the checkout route?", ["backend/app/services/checkout_service.py", "backend/app/routes/checkout.py"]),
        q("Which handler owns POST `/api/refunds`?", ["backend/app/routes/refunds.py"], ["backend/app/services/refund_service.py"]),
        q("Where is the class used as the fake stock repository in tests?", ["backend/tests/inventory_fake.py"]),
        q("Which route delegates token renewal to the authentication service?", ["backend/app/routes/auth.py"], ["backend/app/services/auth_service.py"]),
        q("What repository owns shipment lookup by order ID?", ["backend/app/repositories/shipment_repository.py"], ["backend/app/services/shipping_service.py"]),
        q("Which page imports and calls the checkout API wrapper?", ["frontend/src/pages/CheckoutPage.ts"], ["frontend/src/api/checkout.ts"]),
        q("Where is the admin reimbursement endpoint handler defined?", ["backend/app/routes/admin.py"], ["backend/app/services/refund_service.py"]),
        q("Which utility owns access-token decoding used by authentication middleware?", ["backend/app/utils/jwt_tokens.py"], ["backend/app/middleware/authentication.py"]),
        q("Where is the frontend component that delegates credentials to `signIn`?", ["frontend/src/components/LoginForm.ts"], ["frontend/src/api/auth.ts"]),
        q("Which service method is called by the shipping status route?", ["backend/app/services/shipping_service.py"], ["backend/app/routes/shipping.py"]),
        q("Where is webhook state stored after the payment route accepts an event?", ["backend/app/repositories/payment_repository.py"], ["backend/app/routes/payments.py"]),
        q("Which worker imports stock lookup for reconciliation?", ["worker/jobs/inventory_reconcile.py"], ["backend/app/repositories/inventory_repository.py"]),
        q("Where is the repository definition responsible for changing order status?", ["backend/app/repositories/order_repository.py"]),
        q("Which frontend hook owns refreshing product availability?", ["frontend/src/hooks/useInventory.ts"], ["frontend/src/api/inventory.ts"]),
        q("Where is the backend function called by the profile PATCH handler?", ["backend/app/services/profile_service.py"], ["backend/app/routes/profile.py"]),
    ],
    "relationship": [
        q("Which backend handler receives the request sent when checkout is submitted?", ["frontend/src/api/checkout.ts", "backend/app/routes/checkout.py"]),
        q("What frontend code triggers the endpoint that cancels an order?", ["frontend/src/api/orders.ts", "backend/app/routes/orders.py"]),
        q("Where does the admin refund action eventually reach the backend?", ["frontend/src/api/admin.ts", "backend/app/routes/admin.py"], ["frontend/src/pages/AdminRefund.ts"]),
        q("Which backend route corresponds to the client call used to load an order?", ["frontend/src/api/orders.ts", "backend/app/routes/orders.py"]),
        q("What browser request reaches the login handler?", ["frontend/src/api/auth.ts", "backend/app/routes/auth.py"], ["frontend/src/components/LoginForm.ts"]),
        q("Which server endpoint receives the profile save request?", ["frontend/src/api/profile.ts", "backend/app/routes/profile.py"]),
        q("Connect the card authorization client call to its backend handler.", ["frontend/src/api/payments.ts", "backend/app/routes/payments.py"]),
        q("Which frontend function calls the stock availability endpoint?", ["frontend/src/api/inventory.ts", "backend/app/routes/inventory.py"], ["frontend/src/hooks/useInventory.ts"]),
        q("Where does the stock reservation HTTP call arrive in the backend?", ["frontend/src/api/inventory.ts", "backend/app/routes/inventory.py"]),
        q("Which route handles the browser request for shipment progress?", ["frontend/src/api/shipping.ts", "backend/app/routes/shipping.py"]),
        q("What client function invokes the customer refund request endpoint?", ["frontend/src/api/refunds.ts", "backend/app/routes/refunds.py"]),
        q("Which backend code receives the order submission sent by the web client?", ["frontend/src/api/orders.ts", "backend/app/routes/orders.py"]),
        q("How does the order history page reach the server for one order?", ["frontend/src/pages/OrderHistory.ts", "frontend/src/api/orders.ts", "backend/app/routes/orders.py"]),
        q("Connect session renewal in the hook to the backend refresh endpoint.", ["frontend/src/hooks/useSession.ts", "frontend/src/api/auth.ts", "backend/app/routes/auth.py"]),
        q("Which backend handler answers the UI's current-profile lookup?", ["frontend/src/api/profile.ts", "backend/app/routes/profile.py"]),
        q("Where does `holdInventory` send its request?", ["frontend/src/api/inventory.ts", "backend/app/routes/inventory.py"]),
        q("What server route is paired with `approveRefund`?", ["frontend/src/api/admin.ts", "backend/app/routes/admin.py"]),
        q("Which browser helper and route cooperate to create an order?", ["frontend/src/api/orders.ts", "backend/app/routes/orders.py"]),
    ],
    "hard": [
        q("Which function actually writes a newly created order rather than merely previewing or validating it?", ["backend/app/repositories/order_repository.py"], ["backend/app/services/order_service.py", "frontend/src/utils/orderPreview.ts"]),
        q("Where is stock truly reduced after checkout rather than faked for a test?", ["backend/app/repositories/inventory_repository.py"], ["backend/app/services/inventory_service.py", "backend/tests/inventory_fake.py"]),
        q("What determines the final charged amount instead of only formatting or previewing it?", ["backend/app/services/discount_service.py", "backend/app/services/checkout_service.py"], ["backend/app/utils/money.py", "frontend/src/utils/currency.ts"]),
        q("Where is authentication state validated for protected backend requests rather than created at login?", ["backend/app/middleware/authentication.py"], ["backend/app/services/auth_service.py", "backend/tests/auth_stub.py"]),
        q("What happens after a successful payment provider webhook is received?", ["backend/app/services/payment_gateway.py", "backend/app/repositories/payment_repository.py"], ["backend/app/routes/payments.py"]),
        q("Which implementation records a real authorization rather than returning a mock approval?", ["backend/app/repositories/payment_repository.py"], ["backend/tests/payment_mock.py", "backend/app/services/payment_gateway.py"]),
        q("Where is a cancellation persisted after eligibility is checked?", ["backend/app/repositories/order_repository.py"], ["backend/app/services/order_service.py"]),
        q("Which code creates shipment state rather than simply reading tracking status?", ["backend/app/repositories/shipment_repository.py", "backend/app/services/shipping_service.py"], ["backend/app/routes/shipping.py"]),
        q("Where does an admin approval become a payment reversal?", ["backend/app/services/refund_service.py", "backend/app/repositories/payment_repository.py"], ["backend/app/routes/admin.py"]),
        q("What is the authoritative source of current inventory, not the UI hook or fake repository?", ["backend/app/repositories/inventory_repository.py"], ["frontend/src/hooks/useInventory.ts", "backend/tests/inventory_fake.py"]),
        q("Where are user profile changes actually committed?", ["backend/app/repositories/user_repository.py"], ["backend/app/services/profile_service.py", "frontend/src/api/profile.ts"]),
        q("Which code enforces payment retry exhaustion rather than initially marking a charge for retry?", ["worker/jobs/payment_retry.py"], ["backend/app/repositories/payment_repository.py"]),
        q("Where does the order flow first reserve products and then persist the purchase?", ["backend/app/services/order_service.py"], ["backend/app/services/inventory_service.py", "backend/app/repositories/order_repository.py"]),
        q("Which code is responsible for deduplicating callbacks, not idempotency keys in general?", ["backend/app/repositories/payment_repository.py", "backend/app/services/payment_gateway.py"], ["backend/app/utils/idempotency.py"]),
        q("Where is a refund request stored before any administrator approves it?", ["backend/app/repositories/refund_repository.py"], ["backend/app/services/refund_service.py"]),
        q("Which implementation checks actual credentials rather than returning a test identity?", ["backend/app/services/auth_service.py", "backend/app/repositories/user_repository.py"], ["backend/tests/auth_stub.py"]),
        q("Where is the server-side money rounding behavior, as opposed to browser display formatting?", ["backend/app/utils/money.py"], ["frontend/src/utils/currency.ts"]),
        q("What code releases reserved units when an order is no longer going ahead?", ["backend/app/services/inventory_service.py", "backend/app/services/order_service.py"]),
    ],
}


def build_queries() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for category, examples in QUERIES.items():
        for index, (query, relevance) in enumerate(examples, start=1):
            output.append(
                {
                    "query_id": f"{category}_{index:02d}",
                    "query": query,
                    "category": category,
                    "split": "dev" if index <= 4 else "test",
                    "relevance": relevance,
                }
            )
    return output


def main() -> None:
    for relative_path, content in FILES.items():
        path = REPOSITORY / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    BENCHMARK.write_text(
        json.dumps(build_queries(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
