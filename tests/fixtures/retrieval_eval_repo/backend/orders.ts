import express from "express";

const app = express();

app.post("/api/orders", createOrder);

function createOrder(request, response) {
    const savedOrder = persistPurchase(request.body);
    response.status(201).json(savedOrder);
}

function persistPurchase(orderPayload) {
    return database.orders.insert(orderPayload);
}
