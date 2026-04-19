// scripts/api.js


const api = {
    baseUrl: "http://127.0.0.1:8000",

    // Endpoint 1: Get the general audio summary of the chart
    analyzeChart: async function(imageUrl) {
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            const formData = new FormData();
            formData.append("file", blob, "chart.png");

            const res = await fetch(`${this.baseUrl}/analyze-graph`, {
                method: "POST",
                body: formData
            });
            return await res.text();
        } catch (error) {
            console.error("Error analyzing chart:", error);
            return "Sorry, I could not connect to the STEM Sight vision engine.";
        }
    },

    // Endpoint 2: Ask a specific question using your rule-based NLP
    askChartQuestion: async function(imageUrl, questionText) {
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();
            const formData = new FormData();
            formData.append("file", blob, "chart.png");
            formData.append("question", questionText);

            const res = await fetch(`${this.baseUrl}/ask`, {
                method: "POST",
                body: formData
            });
            return await res.text();
        } catch (error) {
            console.error("Error asking question:", error);
            return "Sorry, I encountered an error while trying to answer that question.";
        }
    }
};

