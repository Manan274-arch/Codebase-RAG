import { approveRefund } from "../api/admin";
export async function confirmRefund(refundId: string) { return approveRefund(refundId); }
