// Constants
const yearElement = document.getElementById('year');
const yearBottomElement = document.getElementById('year-bottom');
const currentYear = new Date().getFullYear();
const menuToggle = document.querySelector('.menu-toggle');
const navbar = document.querySelector('.navbar');
const newsletterForm = document.querySelector('.newsletter-form');

// Footer year
if (yearElement) {
    yearElement.textContent = currentYear;
}
if (yearBottomElement) {
    yearBottomElement.textContent = currentYear;
}

// Mobile menu toggle
document.addEventListener('DOMContentLoaded', function() {
if (menuToggle && navbar) {
    menuToggle.addEventListener('click', function() {
        navbar.classList.toggle('active');
        menuToggle.classList.toggle('active');
        
        // Toggle aria-expanded for accessibility
        const isExpanded = navbar.classList.contains('active');
        menuToggle.setAttribute('aria-expanded', isExpanded);
        
        // Prevent body scroll when menu is open
        if (isExpanded) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    });
    
// Close menu when clicking on a nav link
const navLinks = navbar.querySelectorAll('a');
navLinks.forEach(link => {
    link.addEventListener('click', function() {
        navbar.classList.remove('active');
        menuToggle.classList.remove('active');
        menuToggle.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
    });
});
    
 // Close menu when clicking outside
document.addEventListener('click', function(event) {
    if (!navbar.contains(event.target) && !menuToggle.contains(event.target)) {
        navbar.classList.remove('active');
        menuToggle.classList.remove('active');
        menuToggle.setAttribute('aria-expanded', 'false');
        document.body.style.overflow = '';
    }
});
}

// Newsletter handling for showing message TAKK
if (newsletterForm) {
    newsletterForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const emailInput = this.querySelector('.newsletter-input');
        const submitBtn = this.querySelector('.newsletter-btn');
        
        if (emailInput && emailInput.value) {
            //button text
            const originalText = submitBtn.textContent;
            
            // show message TAKK
            submitBtn.textContent = 'Takk! ✓';
            submitBtn.style.background = 'linear-gradient(135deg, #28a745 0%, #218838 100%)';
            
            // Reset form
            emailInput.value = '';
            
            // Reset button after 3 seconds
            setTimeout(() => {
                submitBtn.textContent = originalText;
                submitBtn.style.background = '';
            }, 3000);
        }
    });
}
});