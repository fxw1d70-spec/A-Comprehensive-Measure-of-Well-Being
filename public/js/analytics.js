// Firebase Analytics.
//
// This apiKey is safe to expose: Firebase web API keys are public identifiers,
// not secrets. They identify the project to Google's servers; access is
// controlled by Firebase Security Rules and API key restrictions in the Google
// Cloud console, not by keeping the key hidden.
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-app.js";
import { getAnalytics, isSupported } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-analytics.js";

const firebaseConfig = {
  apiKey: "AIzaSyDLNnvKCOxnELr6MFNErvr1cbT8weFTy10",
  authDomain: "measure-of-well-being.firebaseapp.com",
  projectId: "measure-of-well-being",
  storageBucket: "measure-of-well-being.firebasestorage.app",
  messagingSenderId: "858394919392",
  appId: "1:858394919392:web:ab9f582fcb3b15ae1dab85",
  measurementId: "G-SWF8KKDCL8",
};

const app = initializeApp(firebaseConfig);

// getAnalytics throws in contexts that lack the APIs it needs (some private
// browsing modes, embedded webviews). Analytics failing must never take the
// prediction UI down with it.
isSupported()
  .then((ok) => {
    if (ok) getAnalytics(app);
  })
  .catch(() => {});
