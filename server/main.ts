import axios, { AxiosResponse } from "axios";
import { AUDIO } from "./large.constant";


interface StatusResponse {
    status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED";
    id: string;
}
(async () => {
    const CONFIG = {
        WORK_ID: "",
        KEY: "",
        REQ_BASE_URL: "https://api.runpod.ai/v2/0zqnqtriady1my"
    }
    const start = Date.now();
    const response = await axios.post<StatusResponse>(`${CONFIG.REQ_BASE_URL}/run`, {

        "input": {
            audio: AUDIO
        }
    },
        {
            headers: {
                Authorization: `Bearer ${CONFIG.KEY}`,
            }
        }
    );
    const statusID = response.data.id;
    CONFIG.WORK_ID = statusID;
    console.log("STATUS ID", statusID)

    
    while (true && CONFIG.WORK_ID) {
        const statusResponse = await axios.get<StatusResponse>(`${CONFIG.REQ_BASE_URL}/status/${CONFIG.WORK_ID}`, {
            headers: {
                Authorization: `Bearer ${CONFIG.KEY}`,
            }
        });
        const status = statusResponse.data;
        console.log("STATUS", status);
        if (status.status === "COMPLETED") {
            console.log("WORK COMPLETED", status);
            break;
        }

        if (status.status === "FAILED") {
            console.log("WORK FAILED", status);
            break;
        }
    }

    const end = Date.now();
    console.log("TOTAL TIME TAKEN", (end - start) / 1000, "seconds");

})()