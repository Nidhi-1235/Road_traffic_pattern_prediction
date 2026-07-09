// script.js

// Mobile menu toggle
const hamburger = document.getElementById("hamburger");
const navLinks = document.getElementById("nav-links");

hamburger.addEventListener("click", () => {
  navLinks.classList.toggle("active");

  // Change hamburger icon to "X" when active
  const icon = hamburger.querySelector("i");
  if (navLinks.classList.contains("active")) {
    icon.classList.remove("fa-bars");
    icon.classList.add("fa-xmark");
  } else {
    icon.classList.remove("fa-xmark");
    icon.classList.add("fa-bars");
  }
});

// Smooth scrolling for nav links
const links = document.querySelectorAll(".nav-links a");

links.forEach(link => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    const targetId = link.getAttribute("href").substring(1);
    const targetSection = document.getElementById(targetId);

    if (targetSection) {
      window.scrollTo({
        top: targetSection.offsetTop - 60, // adjust for navbar height
        behavior: "smooth"
      });
    }

    // Close menu after clicking link (mobile)
    navLinks.classList.remove("active");
    const icon = hamburger.querySelector("i");
    icon.classList.remove("fa-xmark");
    icon.classList.add("fa-bars");
  });
});
