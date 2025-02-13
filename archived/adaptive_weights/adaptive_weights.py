import math

class AdaptiveWeightsAdjustor:
    """
    A class for adaptively adjusting the relative weights of multiple named metrics
    (such as 'recon', 'kl', etc.) using two complementary updates:
      1) Ratio-based update
      2) Responsiveness-based update

    **Key Concepts**:
    - We keep an array of weights w[i] that always sum to 1 (ratio-based).
    - We only update these weights once every `lag` steps. In between, they remain constant.
    - The 'ratio-based' update compares the *slope* (emaFuture - emaPast)/emaPast across metrics.
      This is the usual "ratio_i" approach: w[i] *= exp(eta*(ratio_i - 1)).
    - Immediately after the ratio-based update, we do a 'responsiveness' check:
      Compare how the metric changed (improved or worsened) since the *previous* update, relative
      to how much the weight changed. If the metric responded well, we keep the ratio update as is.
      If it didn't respond (or got worse), we scale down w[i].

    **Lag**:
    - 'lag' is the integer number of steps we wait before re-updating the weights.
    - This means we fix the current w[] for 'lag' steps, presumably to let the model's new weighting
      produce a measurable impact on the metrics. 
    - At each update step (i.e. when step_count % lag == 0), we do:
        ratio_update()
        responsiveness_update()
        store old state for next round

    **EMAs**:
    - We track each metric with two EMAs: emaPast[i], emaFuture[i] using alphaPast, alphaFuture smoothing
      so we can get a stable sense of the metric's 'slope'.

    **Sigmoid** for responsiveness:
    - We measure responsiveness = (old_metric - new_metric)/( new_weight - old_weight ).
    - Then we feed that into a sigmoid, factor = 1/(1 + e^(-alpha * r)), giving (0..1).
    - If r >> 0 => metric improved => factor ~1 => no penalty.
    - If r < 0 => got worse => factor ~0 => we reduce w[i].

    **Usage**:
    1) Construct this class with:
        metric_names=['recon','kl',...],
        w_init={'recon':0.7,'kl':0.3},  # sum=1
        alphaPast, alphaFuture,
        eta, lag, etc.

    2) Each step, call: `step(metrics_dict)`
       This updates the EMAs with the newly observed metrics.
       If current_step % lag==0 (and we're not in warmup), we do both ratio-based & responsiveness updates.
       Otherwise, we do nothing to w[] (weights remain constant).

    3) Access weights via get_weight(metric_name) or manager[ metric_name ].

    This approach ensures the weights remain stable for 'lag' steps,
    hopefully giving time for the new weighting to show an effect in the metrics.
    Then we measure that effect, adjusting further if beneficial or scaling down if not.
    """

    def __init__(
        self,
        metric_names,
        w_init=None,
        alphaPast=0.9,
        alphaFuture=0.5,
        eta=0.1,
        max_weight=1.0,
        master_scale=1.0,
        eps=1e-9,
        lag=1,
        responsiveness_scale=1.0,
        # Let user supply a custom function to map 'responsiveness' => factor,
        # or default to our logistic-based approach (see default_resp_fn).
        compute_responsiveness_factor=None
    ):
        """
        Args:
          metric_names (list[str]): e.g. ['recon','kl',...], the metrics we track.
          w_init (dict|None): initial ratio-based weights. If None => uniform. 
                              Must sum to ~1 if provided.
          alphaPast, alphaFuture (float): smoothing factors for the EMAs of each metric.
                                          (We keep them the same for all metrics here.)
          eta (float): step size for the ratio-based update exponent.
          max_weight (float): clamp w[i] <= this in [0,1]. So no single metric can exceed max_weight.
          master_scale (float): final multiplier for external usage. If 5 => sum of get_all ~5.
          eps (float): small constant for numeric stability.
          lag (int): the number of steps we wait before re-updating. 
                     => We only do an update if (step_count % lag == 0).
          responsiveness_scale (float): exponent scale for applying the responsiveness factor.
          compute_responsiveness_factor (callable or None): optional user-supplied function:
            function signature like: f(responsiveness: float) -> float in (0..1).
            If None, we default to a logistic-based approach (see default_resp_fn).
        """
        self.metric_names = list(metric_names)
        self.name_to_idx = {n: i for i,n in enumerate(self.metric_names)}
        self.N = len(metric_names)

        # 1) Initialize ratio-based weights:
        if w_init is not None:
            self.w = [0.0]*self.N
            for (k,v) in w_init.items():
                idx = self.name_to_idx[k]
                self.w[idx] = v
        else:
            self.w = [1.0/self.N]*self.N
        self.eps = eps

        # 2) Store EMAs
        self.alphaPast   = alphaPast
        self.alphaFuture = alphaFuture
        self.emaPast     = [None]*self.N
        self.emaFuture   = [None]*self.N

        # 3) ratio-based update param
        self.eta = eta

        # 4) bounding & scaling
        self.max_weight = max_weight
        self.master_scale = master_scale

        # 5) lag & responsiveness
        self.lag = lag
        self.responsiveness_scale = responsiveness_scale
        self.compute_resp_fn = compute_responsiveness_factor or self.default_resp_fn
        self._renormalize()

        # 6) We'll store "old" state from the last time we did an update
        #    so that after 'lag' steps, we can measure how the metric changed.
        self.old_w = [None]*self.N
        self.old_m = [None]*self.N

        # 7) internal step counter
        self.step_count = 0

    # ---------------------------------------------------------------------
    # Weighted ratio approach => sum(w)=1
    # ---------------------------------------------------------------------
    def _renormalize(self):
        """
        Ensures sum(w)=1 while also clamping each w[i] in [0, max_weight].
        If sum(w)=0 or very small, fallback to uniform.
        """
        s = sum(self.w)
        if s < self.eps:
            self.w = [1.0/self.N]*self.N
            return
        # clamp each w[i]
        # for i in range(self.N):
        #     if self.w[i] > self.max_weight:
        #         self.w[i] = self.max_weight
        #     if self.w[i] < 0.0:
        #         self.w[i] = 0.0
        # renormalize
        s2 = sum(self.w)
        # if s2 < self.eps:
        #     # fallback
        #     self.w = [1.0/self.N]*self.N
        #     return
        for i in range(self.N):
            self.w[i] /= s2

    # ---------------------------------------------------------------------
    # Updating metric EMAs
    # ---------------------------------------------------------------------
    def update_metric(self, name, value):
        """
        Update the Past & Future EMAs for the specified metric name.
        """
        i = self.name_to_idx[name]
        # If first time, set both
        if self.emaPast[i] is None:
            self.emaPast[i]   = value
            self.emaFuture[i] = value
        else:
            self.emaPast[i]   = self.alphaPast   * self.emaPast[i]   + (1.0 - self.alphaPast)*value
            self.emaFuture[i] = self.alphaFuture * self.emaFuture[i] + (1.0 - self.alphaFuture)*value

    def update_metrics(self, metrics_dict):
        """
        Update multiple metrics from a dict like {'recon':0.123, 'kl':0.456}.
        """
        for (k,v) in metrics_dict.items():
            if k in self.name_to_idx:
                self.update_metric(k, v)

    # ---------------------------------------------------------------------
    # The step function: called once per step in training
    # ---------------------------------------------------------------------
    def step(self, metrics_dict=None):
        """
        - Increments the internal step counter.
        - Optionally update metrics if provided.
        - If step_count % lag==0 (i.e. time to update),
           1) ratio-based update
           2) responsiveness-based update (comparing old vs new state)
           3) store new state as 'old' for next time
        - Otherwise, do nothing => weights remain constant => we let the model adapt to them.
        """
        self.step_count += 1

        # a) update metric EMAs if given
        if metrics_dict is not None:
            self.update_metrics(metrics_dict)

        # b) if not time to update, do nothing
        if self.step_count % self.lag != 0:
            return

        # c) do ratio-based update if we have valid EMAs
        #    i.e. (emaPast[i], emaFuture[i]) not None
        ready_for_ratio = all(m is not None for m in self.emaPast) and all(m is not None for m in self.emaFuture)
        if ready_for_ratio:
            self._ratio_update()

        # d) do responsiveness check if we have "old_w" + "old_m" from the last update
        #    compare them to the newly updated w, metrics => measure if the metric improved
        #    for the weight difference
        if self.old_w[0] is not None:  # or check any
            self._responsiveness_update()

        # e) store new state as old for next time
        self._save_as_old()

    def _ratio_update(self):
        """
        Performs the ratio-based update of weights based on metric improvements.
        
        Steps:
          1. Calculate improvement for each metric.
             improvement_i = -(emaFuture[i] - emaPast[i]) / max(emaPast[i], eps)
             (Assuming lower metric values are better)
          2. Compute the average improvement over all positively improving metrics.
             avg_improvement = (sum of positive improvements) / (number of positive improvements)
             If no metrics are improving, set avg_improvement = eps.
          3. For each metric:
             - If improvement_i > 0:
                 ratio_i = improvement_i / avg_improvement
             - Else:
                 ratio_i = 0
             - Clamp ratio_i to [0, 10] to prevent extreme updates.
             - Compute the update factor:
                 factor = exp(eta * (1 - ratio_i))
             - Update the weight:
                 w[i] *= factor
          4. Renormalize the weights to ensure they sum to 1.
        """
        # 1. Calculate improvements
        improvements = []
        for i in range(self.N):
            improvement_i = -(self.emaFuture[i] - self.emaPast[i]) / max(self.emaPast[i], self.eps)
            improvements.append(improvement_i)

        # 2. Compute average improvement over positively improving metrics
        positive_improvements = [impr for impr in improvements if impr > 0]
        if len(positive_improvements) == 0:
            avg_improvement = self.eps
        else:
            avg_improvement = sum(positive_improvements) / len(positive_improvements)
            avg_improvement = max(avg_improvement, self.eps)  # Ensure it's at least eps

        # 3. Update each weight based on its improvement ratio
        for i in range(self.N):
            if improvements[i] > 0:
                ratio_i = improvements[i] / avg_improvement
            else:
                ratio_i = 0.0  # No improvement or worsening

            # Clamp ratio_i to [0, 10] to prevent extreme updates
            clamped_ratio_i = max(0.0, min(10.0, ratio_i))

            # Compute the update factor
            exponent = self.eta * (1.0 - clamped_ratio_i)
            # Optionally clamp the exponent to avoid overflow in exp()
            exponent = max(-10.0, min(10.0, exponent))
            try:
                factor = math.exp(exponent)
            except OverflowError:
                # If exponent is too large/small, set factor to 0 or a very large number accordingly
                factor = 0.0 if exponent < 0 else float('inf')

            # Update the weight
            self.w[i] *= factor

        # 4. Renormalize weights
        self._renormalize()


    # def _ratio_update(self):
    #     """
    #     Ratio-based update of weights with the desired behavior:
        
    #     For each metric i (assuming a loss where smaller is better), define:
        
    #         improvement_i = - (emaFuture[i] - emaPast[i]) / max(emaPast[i], eps)
        
    #     so that a decreasing (improving) loss gives a positive improvement.
        
    #     Then compute the average improvement:
        
    #         avg_improvement = (1/N) * Σ improvement_i
        
    #     and the ratio:
        
    #         ratio_i = improvement_i / avg_improvement.
        
    #     The desired update is such that:
    #     - If ratio_i > 1 (i.e. the metric is improving faster than average),
    #         then we want to decrease its weight.
    #     - If ratio_i < 1, then we want to increase its weight.
        
    #     We achieve this by updating:
        
    #         w[i] *= exp( eta * (1 - ratio_i) ).
        
    #     We also clamp ratio_i to a reasonable range before computing the exponential,
    #     and finally renormalize so that the weights still sum to 1.
    #     """
    #     slopes = []  # Here "slope" is defined as negative of the usual slope,
    #                 # so that a decrease in the loss becomes a positive improvement.
    #     for i in range(self.N):
    #         denom = max(self.emaPast[i] if self.emaPast[i] is not None else 1.0, self.eps)
    #         # For a loss, improvement is defined as:
    #         # improvement_i = -(emaFuture - emaPast)/denom.
    #         improvement_i = -(self.emaFuture[i] - self.emaPast[i]) / denom
    #         slopes.append(improvement_i)
        
    #     # Compute the average improvement
    #     avg_improvement = sum(slopes) / self.N
    #     avg_improvement = max(avg_improvement, self.eps)  # ensure positive average
        
    #     for i in range(self.N):
    #         ratio_i = slopes[i] / avg_improvement
    #         # Clamp ratio_i to a reasonable range, e.g. [0, 10]
    #         clamped_ratio = max(0, min(10, ratio_i))
            
    #         # Update factor: if clamped_ratio > 1, we want a factor < 1 (decrease weight);
    #         # if clamped_ratio < 1, factor > 1 (increase weight).
    #         # Thus:
    #         exponent = self.eta * (1 - clamped_ratio)
    #         # Optionally clamp the exponent to avoid overflow (e.g., in [-10,10])
    #         exponent = max(-10, min(10, exponent))
    #         factor = math.exp(exponent)
    #         self.w[i] *= factor
        
    #     self._renormalize()


    def _responsiveness_update(self):
        """
        For each metric i:
          delta_w = w[i] - old_w[i]
          improvement = old_m[i] - emaPast[i]   (smaller better => old - new)
          responsiveness = improvement / (delta_w + eps)
          factor = compute_responsiveness_factor(responsiveness)
          w[i] *= factor^responsiveness_scale
        """
        for i in range(self.N):
            delta_w = self.w[i] - self.old_w[i]
            if abs(delta_w) < self.eps:
                # no real weight change => skip
                continue
            improvement = self.old_m[i] - self.emaPast[i]
            r = improvement / delta_w

            factor = self.compute_resp_fn(r)
            # exponent
            self.w[i] *= (factor ** self.responsiveness_scale)

        self._renormalize()

    def _save_as_old(self):
        """
        Save current w[i] and the 'emaPast[i]' as old references,
        so next time we do an update (in lag steps), we can compare.
        """
        for i in range(self.N):
            self.old_w[i] = self.w[i]
            # using emaPast as the baseline 'metric' (assuming smaller=better).
            self.old_m[i] = self.emaPast[i] if self.emaPast[i] is not None else 0.0

    # ---------------------------------------------------------------------
    # Default responsiveness function: logistic
    # ---------------------------------------------------------------------
    def default_resp_fn(self, responsiveness):
        """
        A smooth logistic-based function:
          factor = 1 / (1 + exp(-alpha * responsiveness)),
        with alpha=5 for demonstration.
        => If r >>0 => factor->1,
           if r<<0 => factor->0.
        """
        alpha = 5.0
        exponent = -alpha * responsiveness
        # Clamp the exponent to [-10, 10] to prevent math.exp overflow
        exponent = max(-10, min(10, exponent))
        return 1.0/(1.0 + math.exp(exponent))

    # ---------------------------------------------------------------------
    # External weight access
    # ---------------------------------------------------------------------
    def get_weight(self, metric_name):
        """
        Return the final scaled weight for a given metric, i.e. (master_scale * w[i]).
        """
        i = self.name_to_idx[metric_name]
        return self.master_scale*self.w[i]

    def get_all_weights(self):
        """
        Return a dict: { 'recon': scaledW, 'kl': scaledW, ... }.
        Summation ~ (master_scale).
        """
        out = {}
        for n in self.metric_names:
            out[n] = self.get_weight(n)
        return out

    def __getitem__(self, metric_name):
        """
        Allows usage like manager["recon"] => get_weight("recon").
        """
        return self.get_weight(metric_name)

    def __repr__(self):
        sc_w = {n: f"{self[n]:.4f}" for n in self.metric_names}
        return (f"{self.__class__.__name__}(step_count={self.step_count}, "
                f"lag={self.lag}, scaled_weights={sc_w})")
