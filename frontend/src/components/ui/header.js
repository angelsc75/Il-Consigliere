// Header.js
import React from "react";
import "./header.css"; // Archivo CSS para los estilos

const Header = () => {
  return (
    <header className="app-header">
      <img src="../logo.png" alt="Il Consigliere" className="app-logo" />
      <h1 >Il Consigliere</h1>
    </header>
  );
};

export default Header;
