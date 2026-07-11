# Frontend - Decentralized Recommender Systems

This is the frontend component for the Decentralized Recommender Systems prototype. It provides an interactive user interface, utilizing a smartphone city map layout. 

The original UI design for this project was created in Figma and is available here: [Smartphone City Map Layout](https://www.figma.com/design/eiTkrhnHK5y2pc6JSX5kzi/Smartphone-City-Map-Layout).

## Tech Stack
This project is built using:
- **Vite** (Build tool & development server)
- **React** (UI Library)
- **Tailwind CSS** (Styling)
- **Leaflet & React-Leaflet** (Interactive maps)
- **Radix UI & MUI** (Component libraries)

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
- [Node.js](https://nodejs.org/) (which includes `npm`)

## Getting Started

Follow these steps to set up and run the frontend locally:

1. **Navigate to the frontend directory:**
   Ensure you are in the `frontend` directory of the project.
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   Run the following command to install all required packages:
   ```bash
   npm install
   ```

3. **Start the development server:**
   Launch the Vite development server by running:
   ```bash
   npm run dev
   ```

## Viewing the Frontend

Once the development server is running, the frontend will be accessible in your web browser. 

- Open your browser and navigate to: **[http://localhost:5173](http://localhost:5173)**

*(Note: Make sure the backend API is also running as described in the root documentation so that the frontend can fetch necessary data).*

## How to Use the Prototype

*   **Settings:** 
    *   You can choose which recommendation model you want to use.
    *   You can configure whether to use all users or only a few selected ones (Demo Mode).
*   **Map Markers:**
    *   The **dark blue point** represents the simulated location of the user.
    *   The **light blue point** represents the real check-in. 
    *   If the real check-in is among the recommendations, this point is colored **green**.