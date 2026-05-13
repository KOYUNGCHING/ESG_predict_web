function scrollToModels() {
    const modelSection = document.getElementById("model-section");
    modelSection.scrollIntoView({
        behavior: "smooth"
    });
}


function goToPage(url) {
    window.location.href = url;
}