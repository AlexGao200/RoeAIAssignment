"use client";

import { useState, ChangeEvent } from "react";

const MainPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string>("");
  const [videoUrl, setVideoUrl] = useState<string>("");
  const [videoId, setVideoId] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const [query, setQuery] = useState<string>("");
  const [searchResult, setSearchResult] = useState<any>(null);
  const [searchMessage, setSearchMessage] = useState<string>("");
  const [resultMessage, setResultMessage] = useState<string>("");

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setUploadMessage("");
    setIsUploading(true);
    setVideoUrl("");
    setVideoId("");
    
    const formData = new FormData();
    formData.append("video_file", file);
    formData.append("title", file.name);
    
    try {
      const res = await fetch("http://localhost:8000/api/upload/", {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setUploadMessage("Upload and processing succeeded!");
        setVideoUrl(data.presigned_url || data.file_url || "");
        setVideoId(String(data.video_id));
      } else {
        setUploadMessage("Upload failed.");
      }
    } catch (err) {
      console.error(err);
      setUploadMessage("An error occurred.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!query || !videoId) return;
    setSearchMessage("");
    try {
      const res = await fetch("http://localhost:8000/api/search/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, video_id: videoId }),
      });
      if (res.ok) {
        const data = await res.json();
        if (!data.result) {
          setSearchResult(null);
          setResultMessage("No result found.");
        } else {
          setSearchResult(data.result);
          setSearchMessage("Search successful!");
        }
      } else {
        setSearchMessage("Search failed.");
      }
    } catch (err) {
      console.error(err);
      setSearchMessage("An error occurred during search.");
    }
  };

  return (
    <div className="container">
      <h1>Video Semantic Search</h1>
      
      {/* Upload Section */}
      <section className="section">
        <h2>Upload Video</h2>
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange} 
          className="file-input"
        />
        <button onClick={handleUpload} className="button upload-button">
          Upload
        </button>
        {isUploading && (
          <div className="spinner">
            <p>Uploading and processing... please wait.</p>
          </div>
        )}
        {uploadMessage && <p className="message">{uploadMessage}</p>}
        
        {videoUrl && (
          <div className="video-preview">
            <h3>Video Preview</h3>
            <video width="480" controls>
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        )}
      </section>

      <hr />

      {/* Search Section */}
      <section className="section">
        <h2>Search Video</h2>
        <input
          type="text"
          placeholder="Enter search query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="search-input"
        />
        <button onClick={handleSearch} className="button search-button">
          Search
        </button>
        {searchMessage && <p className="message">{searchMessage}</p>}
        {searchResult ? (
          <div className="search-result">
            <p>
              <strong>
                {parseFloat(searchResult.start).toFixed(2)} - {parseFloat(searchResult.end).toFixed(2)} seconds:
              </strong>{" "}
              {searchResult.text}
            </p>
          </div>
        ) : (
          query && !isUploading && resultMessage === "No result found." && (
            <p className="message">No result found.</p>
          )
        )}
      </section>

      <style jsx>{`
        .container {
          max-width: 800px;
          margin: 0 auto;
          padding: 2rem;
          font-family: Arial, sans-serif;
          color: #333;
          background: #f9f9f9;
        }
        h1 {
          font-size: 30px;
          text-align: center;
          margin-bottom: 10px;
        }
        h2 {
          font-size: 20px;
        }
        h1, h2, h3 {
          color: #222;
          font-weight: bold;
        }
        .section {
          margin-bottom: 2rem;
          background: #fff;
          padding: 1.5rem;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .file-input {
          padding: 0.5rem;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-size: 1rem;
          margin-right: 1rem;
        }
        .search-input {
          width: 100%;
          min-height: 50px;
          padding: 0.5rem;
          border: 1px solid #ccc;
          border-radius: 4px;
          font-size: 1rem;
          margin-right: 1rem;
          resize: vertical;
          white-space: pre-wrap;
        }
        .button {
          padding: 0.6rem 1.2rem;
          border: none;
          border-radius: 4px;
          background-color: #0070f3;
          color: #fff;
          font-size: 1rem;
          cursor: pointer;
          transition: background-color 0.2s;
        }
        .button:hover {
          background-color: #005bb5;
        }
        .upload-button {
          margin-top: 1rem;
        }
        .search-button {
          margin-top: 1rem;
        }
        .spinner {
          margin-top: 1rem;
          font-style: italic;
          color: #555;
        }
        .message {
          margin-top: 1rem;
          font-weight: bold;
        }
        .video-preview, .search-result {
          margin-top: 1rem;
        }
        hr {
          border: none;
          border-top: 1px solid #ddd;
          margin: 2rem 0;
        }
      `}</style>
    </div>
  );
};

export default MainPage;
