import { sendCheckout } from "../api/checkout";
export async function completePurchase(cart: object) { const result = await sendCheckout(cart); return result.data; }
