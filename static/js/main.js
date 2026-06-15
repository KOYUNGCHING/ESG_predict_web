function scrollToModels() {
    const modelSection = document.getElementById("model-section");
    modelSection.scrollIntoView({
        behavior: "smooth"
    });
}


function goToPage(url) {
    window.location.href = url;
}


document.addEventListener("DOMContentLoaded", function () {
    const menuButton = document.querySelector(".hamburger-btn");
    const dropdownMenu = document.querySelector(".dropdown-menu");

    if (!menuButton || !dropdownMenu) {
        return;
    }

    menuButton.addEventListener("click", function () {
        const isOpen = dropdownMenu.classList.toggle("is-open");
        menuButton.setAttribute("aria-expanded", String(isOpen));
    });

    document.addEventListener("click", function (event) {
        const clickedInsideMenu = event.target.closest(".site-menu");

        if (!clickedInsideMenu) {
            dropdownMenu.classList.remove("is-open");
            menuButton.setAttribute("aria-expanded", "false");
        }
    });
});
