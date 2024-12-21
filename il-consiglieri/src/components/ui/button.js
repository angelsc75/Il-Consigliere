import React from "react";
import "./button.css"; // Opcional, para estilos específicos

export function Button({ children, onClick, className, disabled }) {
  return (
    <button
      onClick={onClick}
      className={`button ${className}`}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

