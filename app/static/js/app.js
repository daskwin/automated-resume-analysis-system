const navButtons = document.querySelectorAll(".nav-button");
const views = document.querySelectorAll(".view");

const uploadForm = document.getElementById("uploadForm");
const resumeFile = document.getElementById("resumeFile");
const fileLabel = document.getElementById("fileLabel");
const submitButton = document.getElementById("submitButton");
const errorMessage = document.getElementById("errorMessage");
const matchVacancies = document.getElementById("matchVacancies");

const vacancyForm = document.getElementById("vacancyForm");
const vacancyTitle = document.getElementById("vacancyTitle");
const vacancyCompany = document.getElementById("vacancyCompany");
const vacancyLocation = document.getElementById("vacancyLocation");
const vacancyDescription = document.getElementById("vacancyDescription");
const vacancyRequirements = document.getElementById("vacancyRequirements");
const vacancySubmitButton = document.getElementById("vacancySubmitButton");
const vacancyMessage = document.getElementById("vacancyMessage");
const vacancyList = document.getElementById("vacancyList");
const refreshVacanciesButton = document.getElementById("refreshVacanciesButton");

const resultCard = document.getElementById("resultCard");
const resultFilename = document.getElementById("resultFilename");
const resultTextLength = document.getElementById("resultTextLength");
const textPreview = document.getElementById("textPreview");

const resultCategory = document.getElementById("resultCategory");
const resultModelType = document.getElementById("resultModelType");
const resultConfidenceText = document.getElementById("resultConfidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const topCategories = document.getElementById("topCategories");

const resultItRole = document.getElementById("resultItRole");
const resultItRoleConfidenceText = document.getElementById("resultItRoleConfidenceText");
const itRoleConfidenceBar = document.getElementById("itRoleConfidenceBar");
const topItRoles = document.getElementById("topItRoles");

const matchBlock = document.getElementById("matchBlock");
const vacancyMatches = document.getElementById("vacancyMatches");

const entityEmails = document.getElementById("entityEmails");
const entityPhones = document.getElementById("entityPhones");
const entityLinks = document.getElementById("entityLinks");
const entitySkills = document.getElementById("entitySkills");
const entityPersons = document.getElementById("entityPersons");
const entityDesignations = document.getElementById("entityDesignations");
const entityCompanies = document.getElementById("entityCompanies");
const entityLocations = document.getElementById("entityLocations");
const entityEducation = document.getElementById("entityEducation");


document.addEventListener("DOMContentLoaded", () => {
    loadVacancies();
});


navButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const viewId = button.dataset.view;

        navButtons.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");

        views.forEach((view) => {
            if (view.id === viewId) {
                view.classList.add("active-view");
            } else {
                view.classList.remove("active-view");
            }
        });

        if (viewId === "vacanciesView") {
            loadVacancies();
        }
    });
});


resumeFile.addEventListener("change", () => {
    if (resumeFile.files.length > 0) {
        fileLabel.textContent = resumeFile.files[0].name;
    } else {
        fileLabel.textContent = "Выберите файл резюме";
    }
});


uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideError();

    if (!resumeFile.files || resumeFile.files.length === 0) {
        showError("Выберите файл резюме.");
        return;
    }

    const formData = new FormData();
    formData.append("file", resumeFile.files[0]);
    formData.append("match_vacancies", matchVacancies.checked ? "true" : "false");

    submitButton.disabled = true;
    submitButton.textContent = "Анализируем...";

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Не удалось выполнить анализ.");
        }

        renderResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Анализировать";
    }
});


vacancyForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    hideVacancyMessage();

    const formData = new FormData();
    formData.append("title", vacancyTitle.value.trim());
    formData.append("company", vacancyCompany.value.trim());
    formData.append("location", vacancyLocation.value.trim());
    formData.append("description", vacancyDescription.value.trim());
    formData.append("requirements", vacancyRequirements.value.trim());

    vacancySubmitButton.disabled = true;
    vacancySubmitButton.textContent = "Сохраняем...";

    try {
        const response = await fetch("/api/vacancies", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Не удалось сохранить вакансию.");
        }

        vacancyForm.reset();
        showVacancyMessage("Вакансия сохранена.");
        loadVacancies();
    } catch (error) {
        showVacancyMessage(error.message, true);
    } finally {
        vacancySubmitButton.disabled = false;
        vacancySubmitButton.textContent = "Сохранить";
    }
});


refreshVacanciesButton.addEventListener("click", () => {
    loadVacancies();
});


async function loadVacancies() {
    try {
        const response = await fetch("/api/vacancies");
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Не удалось загрузить вакансии.");
        }

        renderVacancyList(data.vacancies || []);
    } catch (error) {
        vacancyList.innerHTML = `<p class="error-text">${escapeHtml(error.message)}</p>`;
    }
}


function renderVacancyList(vacancies) {
    vacancyList.innerHTML = "";

    if (!vacancies || vacancies.length === 0) {
        vacancyList.innerHTML = "<p class='muted'>Вакансии пока не добавлены.</p>";
        return;
    }

    vacancies.forEach((vacancy) => {
        const item = document.createElement("article");
        item.className = "vacancy-item";

        item.innerHTML = `
            <div class="vacancy-item-top">
                <div>
                    <div class="vacancy-title-row">
                        <strong>${escapeHtml(vacancy.title || "-")}</strong>
                        <span>#${vacancy.id}</span>
                    </div>
                    <p>${escapeHtml(vacancy.company || "Компания не указана")} · ${escapeHtml(vacancy.location || "Локация не указана")}</p>
                </div>

                <button type="button" class="delete-button" data-id="${vacancy.id}">
                    Удалить
                </button>
            </div>

            <div class="vacancy-text">
                ${escapeHtml(vacancy.description || "")}
            </div>

            <div class="vacancy-requirements">
                ${escapeHtml(vacancy.requirements || "")}
            </div>
        `;

        vacancyList.appendChild(item);
    });

    document.querySelectorAll(".delete-button").forEach((button) => {
        button.addEventListener("click", async () => {
            const vacancyId = button.dataset.id;
            await deleteVacancy(vacancyId);
        });
    });
}


async function deleteVacancy(vacancyId) {
    try {
        const response = await fetch(`/api/vacancies/${vacancyId}`, {
            method: "DELETE",
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Не удалось удалить вакансию.");
        }

        loadVacancies();
    } catch (error) {
        showVacancyMessage(error.message, true);
    }
}


function renderResult(data) {
    resultCard.classList.remove("hidden");

    resultFilename.textContent = data.filename;
    resultTextLength.textContent = `${data.text_length} символов`;
    textPreview.textContent = data.text_preview || "Текст не найден";

    renderVacancyMatches(data);
    renderGeneralDomain(data);
    renderItRole(data);
    renderEntities(data);
}


function renderVacancyMatches(data) {
    const matches = data.vacancy_matches || [];

    if (!matches.length) {
        matchBlock.classList.add("hidden");
        vacancyMatches.innerHTML = "";
        return;
    }

    matchBlock.classList.remove("hidden");
    vacancyMatches.innerHTML = "";

    matches.slice(0, 5).forEach((match, index) => {
        const percent = Math.round(match.match_percent || 0);
        const matchedSkills = match.matched_skills || [];
        const missingSkills = match.missing_skills || [];

        const item = document.createElement("article");
        item.className = "match-item";

        item.innerHTML = `
            <div class="match-header">
                <div>
                    <div class="match-title-line">
                        <span class="match-rank">#${index + 1}</span>
                        <strong>${escapeHtml(match.title || "-")}</strong>
                    </div>
                    <p>${escapeHtml(match.company || "Компания не указана")} · ${escapeHtml(match.location || "Локация не указана")}</p>
                </div>
                <span class="match-score">${percent}%</span>
            </div>

            <div class="progress match-progress">
                <div class="progress-bar" style="width: ${percent}%"></div>
            </div>

            <p class="match-explanation">${escapeHtml(cleanExplanation(match.explanation || ""))}</p>

            <div class="match-score-grid">
                <div>
                    <span>Навыки</span>
                    <strong>${Math.round((match.skill_match_score || 0) * 100)}%</strong>
                </div>
                <div>
                    <span>Контекст</span>
                    <strong>${Math.round((match.semantic_score || 0) * 100)}%</strong>
                </div>
                <div>
                    <span>Роль</span>
                    <strong>${Math.round((match.role_match_score || 0) * 100)}%</strong>
                </div>
            </div>

            <div class="match-skills">
                <span>Совпавшие навыки</span>
                <div class="entity-list">
                    ${renderPillsHtml(matchedSkills)}
                </div>
            </div>

            <div class="match-skills">
                <span>Недостающие навыки</span>
                <div class="entity-list">
                    ${renderPillsHtml(missingSkills)}
                </div>
            </div>
        `;

        vacancyMatches.appendChild(item);
    });
}


function cleanExplanation(value) {
    return value
        .replaceAll("Предсказанная IT-роль кандидата:", "Специализация кандидата:")
        .replaceAll("Совпавшие навыки:", "Совпали навыки:")
        .replaceAll("Недостающие навыки:", "Не хватает:")
        .replaceAll("Вакансия:", "Позиция:");
}


function renderGeneralDomain(data) {
    const domain = data.predicted_domain || data.predicted_category || "-";
    const topDomains = data.top_3_domains || data.top_3_categories || [];

    resultCategory.textContent = domain;

    if (data.model_type === "embedding" && data.embedding_model) {
        resultModelType.textContent = `${data.model_type}: ${data.embedding_model} + ${data.classifier_name || ""}`;
    } else {
        resultModelType.textContent = `${data.model_type || "tfidf"} + ${data.classifier_name || "LogisticRegression"}`;
    }

    const confidencePercent = Math.round((data.confidence || 0) * 100);

    resultConfidenceText.textContent = `${confidencePercent}%`;
    confidenceBar.style.width = `${confidencePercent}%`;

    topCategories.innerHTML = "";

    topDomains.forEach((item) => {
        const percent = Math.round(item.score * 100);

        const div = document.createElement("div");
        div.className = "top-item";
        div.innerHTML = `
            <span>${escapeHtml(item.category)}</span>
            <strong>${percent}%</strong>
        `;

        topCategories.appendChild(div);
    });
}


function renderItRole(data) {
    if (!data.it_role) {
        return;
    }

    resultItRole.textContent = data.it_role.predicted_role;

    const itConfidencePercent = Math.round((data.it_role.confidence || 0) * 100);

    resultItRoleConfidenceText.textContent = `${itConfidencePercent}%`;
    itRoleConfidenceBar.style.width = `${itConfidencePercent}%`;

    topItRoles.innerHTML = "";

    const topRoles = data.it_role.top_roles || [];

    topRoles.slice(0, 3).forEach((item) => {
        const percent = Math.round(item.score * 100);

        const div = document.createElement("div");
        div.className = "top-item";
        div.innerHTML = `
            <span>${escapeHtml(item.role)}</span>
            <strong>${percent}%</strong>
        `;

        topItRoles.appendChild(div);
    });
}


function renderEntities(data) {
    if (!data.entities) {
        return;
    }

    renderEntityList(entityEmails, data.entities.emails);
    renderEntityList(entityPhones, data.entities.phones);
    renderEntityList(entityLinks, data.entities.links);
    renderEntityList(entitySkills, data.entities.skills);
    renderEntityList(entityPersons, data.entities.person_names);
    renderEntityList(entityDesignations, data.entities.designations);
    renderEntityList(entityCompanies, data.entities.companies);
    renderEntityList(entityLocations, data.entities.locations);
    renderEntityList(entityEducation, data.entities.education);
}


function renderEntityList(element, values) {
    if (!element) {
        return;
    }

    if (!values || values.length === 0) {
        element.innerHTML = "<span class='muted'>Не найдено</span>";
        return;
    }

    element.innerHTML = "";

    values.forEach((value) => {
        const item = document.createElement("span");
        item.className = "entity-pill";
        item.textContent = value;
        element.appendChild(item);
    });
}


function renderPillsHtml(values) {
    if (!values || values.length === 0) {
        return "<span class='muted'>Нет</span>";
    }

    return values
        .slice(0, 12)
        .map((value) => `<span class="entity-pill">${escapeHtml(value)}</span>`)
        .join("");
}


function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
}


function hideError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
}


function showVacancyMessage(message, isError = false) {
    vacancyMessage.textContent = message;
    vacancyMessage.classList.remove("hidden");

    if (isError) {
        vacancyMessage.classList.add("error-like");
    } else {
        vacancyMessage.classList.remove("error-like");
    }
}


function hideVacancyMessage() {
    vacancyMessage.textContent = "";
    vacancyMessage.classList.add("hidden");
    vacancyMessage.classList.remove("error-like");
}


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
