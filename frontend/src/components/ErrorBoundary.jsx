import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("UI error", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="fatal-error">
        <h2>Something went wrong in the page</h2>
        <p>{String(this.state.error.message || this.state.error)}</p>
        <button className="primary-button" type="button" onClick={() => window.location.reload()}>Reload</button>
      </div>
    );
  }
}
