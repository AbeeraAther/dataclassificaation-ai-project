const form = document.getElementById("prediction-form");
const predictBtn = document.getElementById("predict-btn");
const errorBox = document.getElementById("error-box");

const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const resultSpecies = document.getElementById("result-species");
const resultConfidence = document.getElementById("result-confidence");
const resultIcon = document.getElementById("result-icon");
const probabilityBars = document.getElementById("probability-bars");

const SPECIES_ICONS = {
    "Iris Setosa": "🌱",
    "Iris Versicolor": "🌷",
    "Iris Virginica": "🌺"
};

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    hideError();

    const sepalLength = document.getElementById("sepal_length").value.trim();
    const sepalWidth = document.getElementById("sepal_width").value.trim();
    const petalLength = document.getElementById("petal_length").value.trim();
    const petalWidth = document.getElementById("petal_width").value.trim();

    const fields = { sepalLength, sepalWidth, petalLength, petalWidth };

    for (const [name, value] of Object.entries(fields)) {
        if (value === "") {
            showError("Please fill in all four measurement fields.");
            return;
        }
        const numericValue = Number(value);
        if (Number.isNaN(numericValue) || numericValue <= 0) {
            showError("All measurements must be positive numbers.");
            return;
        }
    }

    const requestData = {
        sepal_length: sepalLength,
        sepal_width: sepalWidth,
        petal_length: petalLength,
        petal_width: petalWidth
    };

    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || "Something went wrong. Please try again.");
            return;
        }

        displayResult(data);

    } catch (error) {
        showError("Could not reach the server. Please make sure the Flask app is running.");
    } finally {
        setLoading(false);
    }
});

function displayResult(data) {
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    resultSpecies.textContent = data.prediction;
    resultConfidence.textContent = data.confidence;
    resultIcon.textContent = SPECIES_ICONS[data.prediction] || "🌼";

    probabilityBars.innerHTML = "";
    for (const [species, probability] of Object.entries(data.probabilities)) {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
            <div class="prob-label">
                <span>${species}</span>
                <span>${probability}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${probability}%;"></div>
            </div>
        `;
        probabilityBars.appendChild(row);
    }
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function hideError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
}

function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    predictBtn.textContent = isLoading ? "Predicting..." : "Predict Species";
}
