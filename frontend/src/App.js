import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [tabs, setTabs] = useState([]);
  const [categories, setCategories] = useState({});

  useEffect(() => {
    // Fetch tabs from Chrome API
    if (typeof chrome !== 'undefined' && chrome.tabs) {
      chrome.tabs.query({}, (result) => {
        setTabs(result);
      });
    }
  }, []);

  const categorizeTabs = async () => {
    const tabData = tabs.map(tab => ({
      title: tab.title,
      url: tab.url
    }));

    try {
      const response = await fetch('/categorize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(tabData)
      });
      const data = await response.json();
      setCategories(data.categories);
    } catch (error) {
      console.error('Error categorizing tabs:', error);
    }
  };

  return (
    <div className="App">
      <h1>Intelligent Tab Organizer</h1>
      <button onClick={categorizeTabs}>Categorize Tabs</button>
      {Object.entries(categories).map(([category, tabIndices]) => (
        <div key={category}>
          <h2>{category}</h2>
          <ul>
            {tabIndices.map(index => (
              <li key={index}>{tabs[index].title}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export default App;
