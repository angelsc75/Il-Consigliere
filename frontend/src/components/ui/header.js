// Header.js
import React from "react";
import "./header.css"; // Archivo CSS para los estilos

const Header = ({ userId, onLogout }) => {
  return (
    <header className="app-header">
  <img src="../logo.png" alt="Il Consigliere" className="app-logo" />
  <h1>Il Consigliere</h1>
  <div className="header-actions">
    {userId && (
      <div className="user-info">
        <span className="user-id">User ID: {userId}</span>
        <button className="logout-button" onClick={onLogout}>
          Salir
        </button>
      </div>
    )}
  </div>
</header>
  );
};

export default Header;

