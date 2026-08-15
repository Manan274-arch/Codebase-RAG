import { loadAvailability } from "../api/inventory";
export async function refreshStock(sku: string) { return loadAvailability(sku); }
