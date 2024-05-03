import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.0.2/firebase-app.js';
import { getAuth, signInWithEmailAndPassword, createUserWithEmailAndPassword, sendEmailVerification, sendPasswordResetEmail, setPersistence, browserLocalPersistence } from 'https://www.gstatic.com/firebasejs/9.0.2/firebase-auth.js';


// Your Firebase configuration
// const firebaseConfig = {
//     apiKey: "AIzaSyCFn92ld5RXyzfyKoO9MLx6lrWIVVlSW_Y",
//     authDomain: "speakeridentificationsystem.firebaseapp.com",
//     projectId: "speakeridentificationsystem",
//     storageBucket: "speakeridentificationsystem.appspot.com",
//     messagingSenderId: "472860465046",
//     appId: "1:472860465046:web:ffbbf188f4ab9a21caafb7",
//     measurementId: "G-T1K75N260B"
// };
const firebaseConfig = {
    apiKey: "AIzaSyDx6w33VvYCQZ7LrFbEMkb0977pSJsUobc",
    authDomain: "speaker-identification-system.firebaseapp.com",
    projectId: "speaker-identification-system",
    storageBucket: "speaker-identification-system.appspot.com",
    messagingSenderId: "733461261375",
    appId: "1:733461261375:web:56efe66791cc379cfc9c8a",
    measurementId: "G-2G3KE344MS"
};

// Initialize Firebase and set up session persistence
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
setUpSessionPersistence();



// Function to show loader
function showLoader() {
    document.querySelector('.loader').style.display = 'block';
}

// Function to hide loader
function hideLoader() {
    document.querySelector('.loader').style.display = 'none';
}

// Function to validate email format
function validateEmail(email) {
    const re = /\S+@\S+\.\S+/;
    return re.test(email);
}

// Function to validate password length
function validatePassword(password) {
    return password.length >= 6; // Change the minimum password length as needed
}

// Function to show the logout modal
function showLogoutModal() {
    // Check if the user is logged in
    if (auth.currentUser) {
        // Show user email and user name
        // Fetch user details from Flask backend
        fetch('/user_details/' + auth.currentUser.email)  // Concatenate the user_email variable
            .then(response => response.json())
            .then(data => {
                // Update user name in the Dashboard modal
                document.getElementById('userName').textContent = data.first_name + ' ' + data.last_name;
                // Update user email in the Dashboard modal
                document.getElementById('userEmail').textContent = data.email;
            })
            .catch(error => {
                console.error('Error:', error);
            });
        // Show logout modal
        $('#logoutModal').modal('show');
    }
}

let logoutTimer; // Variable to store the logout timer

// Function to start the logout timer
function startLogoutTimer() {
    // Clear existing timer if any
    if (logoutTimer) {
        clearTimeout(logoutTimer);
    }

    // Set a new timer for 10 minutes (600000 milliseconds)
    logoutTimer = setTimeout(() => {
        // Call logout function after 10 minutes of inactivity
        logoutUser2();
    }, 600000); // 10 minutes in milliseconds
}

// Function to reset the logout timer on user activity
function resetLogoutTimer() {
    startLogoutTimer(); // Restart the timer on user activity
}

// Add event listeners to detect user activity and reset the logout timer
document.addEventListener('mousemove', resetLogoutTimer);
document.addEventListener('keypress', resetLogoutTimer);
document.addEventListener('click', resetLogoutTimer);

// Function to update UI after successful login
function updateUIAfterLogin(user) {
    // Close login modal
    $('#loginModal').modal('hide');
    // Show welcome message with user's name
    document.getElementById('welcomeMessage').innerText = 'Welcome, ' + user.email;
    // Hide login and signup links
    document.getElementById('loginLink').style.display = 'none';
    document.getElementById('signupLink').style.display = 'none';
    document.getElementById('modalLogout').style.display = 'block';
    // // Show logout modal
    // $('#logoutModal').modal('show');
    // Start the logout timer
    startLogoutTimer();
}
// Function to set up session persistence
function setUpSessionPersistence() {
    // Set persistence to 'LOCAL'
    setPersistence(auth, browserLocalPersistence)
        .then(() => {
            console.log("Session persistence set to LOCAL");
            // console.log(auth.currentUser.email);
            // Check if there is a signed-in user
            if (auth.currentUser) {
                // Call updateUIAfterLogin with the signed-in user
                updateUIAfterLogin(auth.currentUser);
            }

        })
        .catch((error) => {
            console.error("Error setting persistence:", error);
        });
}
// Function to start the logout timer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    startLogoutTimer();
});


// Login function
function loginUser() {
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    // Validate email and password
    if (!validateEmail(email)) {
        console.error('Invalid email format');
        // You can display an error message to the user
        return;
    }
    if (!validatePassword(password)) {
        console.error('Password must be at least 6 characters long');
        // You can display an error message to the user
        return;
    }

    // Show loader
    showLoader();

    signInWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Signed in
            const user = userCredential.user;
            if (!user.emailVerified) {
                // If email is not verified, show a message and sign out the user
                alert('Email is not verified. Please verify your email before logging in.');
                // Hide loader
                hideLoader();
                return auth.signOut();
            }
            // console.log('User logged in:', user);
            // alert('Login Successfully !....')
            // Clear email and password fields
            document.getElementById('loginEmail').value = '';
            document.getElementById('loginPassword').value = '';
            // Update UI
            updateUIAfterLogin(user);
            // Hide loader
            hideLoader();
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error('Login error:', errorMessage);

            // Handle specific error cases
            if (errorCode === 'auth/user-not-found') {
                alert('User not found. Please check your email or sign up.');
            } else if (errorCode === 'auth/wrong-password') {
                alert('Incorrect password. Please try again.');
            } else if (errorCode === 'auth/invalid-login-credentials') {
                alert('Invalid login credentials. Please check your email and password.');
            } else {
                alert('An error occurred. Please try again later.');
            }

            // Hide loader
            hideLoader();
        });
}

document.querySelector('#loginModal form').addEventListener('submit', function (event) {
    event.preventDefault();
    loginUser();
});

// Signup function
function signupUser() {
    const firstName = document.getElementById('signupFirstName').value;
    const lastName = document.getElementById('signupLastName').value;
    const email = document.getElementById('signupEmail').value;
    const password = document.getElementById('signupPassword').value;

    // Validate email and password
    if (!validateEmail(email)) {
        console.error('Invalid email format');
        alert('Invalid Email Format');
        // You can display an error message to the user
        return;
    }
    if (!validatePassword(password)) {
        console.error('Password must be at least 6 characters long');
        alert('Password must be at least 6 characters long');
        // You can display an error message to the user
        return;
    }

    // Show loader
    showLoader();

    createUserWithEmailAndPassword(auth, email, password)
        .then((userCredential) => {
            // Signed up
            const user = userCredential.user;
            // console.log('User signed up:', user);
            hideLoader();

            sendVerificationEmail(user);

            // Close signup modal
            $('#signupModal').modal('hide');

            // Show login modal
            $('#loginModal').modal('show');

            // Clear email and password fields
            document.getElementById('signupFirstName').value = '';
            document.getElementById('signupLastName').value = '';
            document.getElementById('signupEmail').value = '';
            document.getElementById('signupPassword').value = '';

            // Hide loader
            hideLoader();
        })
        .catch((error) => {
            const errorCode = error.code;
            const errorMessage = error.message;
            console.error('Signup error:', errorMessage);

            // Handle specific error cases
            if (errorCode === 'auth/email-already-in-use') {
                alert('Email is already registered. Please login or use a different email.');
            } else {
                alert('An error occurred. Please try again later.');
            }

            // Hide loader
            hideLoader();
        });
}

// Method to send the user verification email
async function sendVerificationEmail(user) {
    try {
        // Define action code settings
        const actionCodeSettings = {
            url: 'https://speakeridentificationsystem.netlify.app/', // Your website URL
            handleCodeInApp: true,
            // Additional platform-specific settings can be added here if needed
        };

        // Send email verification using action code settings
        await sendEmailVerification(user, actionCodeSettings);

        // Email verification sent successfully
        alert('Email verification sent! Check your mailbox.');
    } catch (error) {
        // Handle any errors that occurred during email verification
        alert('Error sending email verification:');
    }
}



document.querySelector('#signupModal form').addEventListener('submit', function (event) {
    event.preventDefault();
    signupUser();
});
document.getElementById('signupForm').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the default form submission

    // Show loader
    document.querySelector('.loader-container').style.display = 'flex';

    // Get form data
    const formData = new FormData(this);

    // Send form data to Flask endpoint using fetch
    fetch('/signup', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            // Hide loader
            document.querySelector('.loader-container').style.display = 'none';

            // Handle response
            alert(data.message); // You can replace this with your desired action (e.g., showing a success message or redirecting)
        })
        .catch(error => {
            // Hide loader
            document.querySelector('.loader-container').style.display = 'none';

            console.error('Error:', error);
        });
});



document.querySelector('#modalLogout').addEventListener('click', function (event) {
    event.preventDefault();
    showLogoutModal();
});
document.querySelector('#logout').addEventListener('click', function (event) {
    event.preventDefault();
    logoutUser();
});
// Logout function
function logoutUser() {
    // Show confirmation alert
    if (confirm('Are you sure you want to logout?')) {
        // Proceed with logout
        auth.signOut().then(() => {
            // Sign-out successful.
            console.log('User signed out');
            // Hide logout modal
            $('#logoutModal').modal('hide');
            // Show login and signup links
            document.getElementById('loginLink').style.display = 'block';
            document.getElementById('signupLink').style.display = 'block';
            document.getElementById('modalLogout').style.display = 'none';
            // Clear welcome message
            document.getElementById('welcomeMessage').innerText = '';
        }).catch((error) => {
            // An error happened.
            console.error('Logout error:', error);
        });
    }
}

// Logout function
function logoutUser2() {
    // Proceed with logout
    auth.signOut().then(() => {
        // Sign-out successful.
        console.log('User signed out');
        // Hide logout modal
        $('#logoutModal').modal('hide');
        // Show login and signup links
        document.getElementById('loginLink').style.display = 'block';
        document.getElementById('signupLink').style.display = 'block';
        document.getElementById('modalLogout').style.display = 'none';
        // Clear welcome message
        document.getElementById('welcomeMessage').innerText = '';
    }).catch((error) => {
        // An error happened.
        console.error('Logout error:', error);
    });

}


// Function to handle "Forgot Password?" button click
document.querySelector('#forgotPassword').addEventListener('click', function (event) {
    // Show the forgot password modal
    $('#forgotPasswordModal').modal('show');
    // Hide login modal
    $('#loginModal').modal('hide');
});

// Function to handle forgot password form submission
document.getElementById('forgotPasswordForm').addEventListener('submit', function (event) {
    event.preventDefault();

    // Get the email entered by the user
    const email = document.getElementById('forgotPasswordEmail').value;

    // Call a function to handle the password reset process
    resetPassword(email);
});

// Function to reset the password
function resetPassword(email) {
    console.log('reset function called');
    // Implement your password reset logic here
    // For example, you can use Firebase Authentication's password reset feature:
    // Call sendPasswordResetEmail function with the auth instance, user's email

    // Call the function
    sendPasswordResetEmail(auth, email)
        .then(() => {
            // Password reset email sent successfully
            alert('Password reset email sent');
            // Hide the forgot password modal
            $('#forgotPasswordModal').modal('hide');
        })
        .catch((error) => {
            // Error occurred while sending password reset email
            alert('Error sending password reset email:');
            // Hide the forgot password modal
            $('#forgotPasswordModal').modal('hide');
        });

}
// Function to check if the user is logged in
function isLoggedIn() {
    // console.log("function called")
    // console.log(auth.currentUser)

    // Check if 'auth' is defined and if the user is logged in
    if (typeof auth !== 'undefined' && auth.currentUser) {
        return true;
    } else {
        return false;
    }
}


// Function to handle the click event on the "Record Voice" button
function recordVoiceClicked() {
    if (isLoggedIn()) {
        // User is logged in, navigate to the record voice page
        window.location.href = "/record_train";
        // console.log("User logged in, navigating to record train page");
    } else {
        // User is not logged in, display an alert and open the login modal
        alert("Please log in to access this feature.");
        $('#loginModal').modal('show'); // Open the login modal
    }
}

function recordVoiceClicked2() {
    if (isLoggedIn()) {
        // User is logged in, navigate to the record voice page
        window.location.href = "/record_test";
        // console.log("User logged in, navigating to record train page");
    } else {
        // User is not logged in, display an alert and open the login modal
        alert("Please log in to access this feature.");
        $('#loginModal').modal('show'); // Open the login modal
    }
}


// Attach click event handlers to elements with IDs 'logincheck' and 'logincheck2'
document.getElementById('logincheck').addEventListener('click', recordVoiceClicked);
document.getElementById('logincheck2').addEventListener('click', recordVoiceClicked2);



