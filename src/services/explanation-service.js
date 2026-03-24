export class ExplanationService {
  explain({ snapshot, signal, decision, backtestSummary, riskAssessment }) {
    const reasons = [
      `${snapshot.symbol} is trading at ${snapshot.quote.price} with ${snapshot.quote.changePct}% change.`,
      `Trend is ${decision.indicators.trend}, momentum is ${decision.indicators.momentum}, and volume ratio is ${decision.indicators.volumeRatio}.`,
      `The setup is classified as ${signal.setup} on the ${signal.timeframe} timeframe.`,
      `Confidence is ${decision.confidence} and the risk engine marked execution as ${riskAssessment.executionMode}.`,
      `Recent replay shows ${backtestSummary.totalReturnPct}% strategy return with ${backtestSummary.winRatePct}% hit rate across ${backtestSummary.bars} bars.`
    ];

    if (riskAssessment.reasons.length > 0) {
      reasons.push(`Risk controls: ${riskAssessment.reasons.join(" ")}`);
    }

    return reasons.join(" ");
  }
}
