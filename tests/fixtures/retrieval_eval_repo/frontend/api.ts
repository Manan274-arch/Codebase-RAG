import axios from "axios";

export async function submitOrder(items: string[]) {
    return axios.post("/api/orders", { items });
}

export async function loadPaymentStatus(invoiceId: string) {
    return axios.get(`/api/invoices/${invoiceId}/payment-status`);
}
