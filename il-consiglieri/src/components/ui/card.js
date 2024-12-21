import React from "react";
import "./card.css"; // Opcional, para estilos específicos

export function Card({ children, className }) {
  return <div className={`card ${className}`}>{children}</div>;
}

export function CardContent({ children }) {
  return <div className="card-content">{children}</div>;
}

export function CardDescription({ children }) {
  return <p className="card-description">{children}</p>;
}

export function CardFooter({ children }) {
  return <div className="card-footer">{children}</div>;
}

export function CardHeader({ children }) {
  return <div className="card-header">{children}</div>;
}

export function CardTitle({ children }) {
  return <h3 className="card-title">{children}</h3>;
}
