## 1. **CBF Backup Operator (Theorem 4.1)**

### **Equation:**
\[
\Phi_{\text{FISOR}}(x, y) = (1-\gamma)x + \gamma \min\{x, y\}
\]

### **Code Implementation:**
```python
def phi_fisor(x, y, gamma):
    # Implements Φ_FISOR(x, y) = (1-γ)x + γ min{x, y}
    return (1 - gamma) * x + gamma * jnp.minimum(x, y)
```
- **Location:** Top of cbf.py
- **Implements:** The backup operator Φ for cost critic updates.

---

## 2. **CBF Cost Critic Update (Eq. 1, 2, 3, 4 from formulation)**

### **Equations:**
- \( V_h^*(s) = \Phi(h(s), \max_a V_h^*(f(s,a))) \)
- \( Q^h(s,a) \leftarrow \max\{h(s), V^h(f(s,a))\} \)
- Expectile regression for value update.

### **Code Implementation:**
```python
def cbf_loss_fn(self, cbf_params, batch):
    h_s = batch["costs"]  # h(s)
    next_v = self.safe_value.apply_fn({"params": self.safe_value.params}, batch["next_observations"])
    phi = phi_fisor(h_s, next_v, self.cbf_gamma)
    qcs = self.safe_critic.apply_fn({"params": cbf_params}, batch["observations"], batch["actions"])
    loss = ((qcs - phi) ** 2).mean()
    # Admissibility penalty (Theorem 4.1)
    admissibility_violation = jnp.maximum(phi - h_s, 0).mean()
    total_loss = loss + self.cbf_admissibility_coef * admissibility_violation
    return total_loss, {"cbf_loss": loss, "admissibility_violation": admissibility_violation}
```
- **Location:** Method `cbf_loss_fn` in `CBF` class.
- **Implements:** Cost critic update using FISOR backup operator and admissibility penalty.

---

## 3. **Reward Critic Piecewise Target (Eq. 1, 2 from reward_critic)**

### **Equations:**
\[
Q^R_{\text{target}}(s,a,s') = 
\begin{cases}
r(s,a) + \gamma V^R(s') & \text{if } Q^h(s,a) \leq 0 \\
\frac{r_{\min}}{1-\gamma} - Q^h(s,a) & \text{if } Q^h(s,a) > 0
\end{cases}
\]
\[
V^R_{\text{target}}(s) = 
\begin{cases}
Q^R(s,a) & \text{if } Q^h(s,a) \leq 0 \\
\frac{r_{\min}}{1-\gamma} - Q^h(s,a) & \text{if } Q^h(s,a) > 0
\end{cases}
\]

### **Code Implementation:**
```python
def reward_piecewise_target(self, batch):
    qh = self.safe_critic.apply_fn({"params": self.safe_critic.params}, batch["observations"], batch["actions"]).max(axis=0)
    next_vr = self.value.apply_fn({"params": self.value.params}, batch["next_observations"])
    mask_unsafe = (qh > 0)
    mask_safe = (qh <= 0)
    target = (
        mask_safe * (batch["rewards"] + self.discount * batch["masks"] * next_vr)
        + mask_unsafe * (self.r_min / (1 - self.discount) - qh)
    )
    return target
```
- **Location:** Method `reward_piecewise_target` in `CBF` class.
- **Implements:** Piecewise target for reward critic.

---

## 4. **Reward Critic Loss (Eq. 3, 4 from formulation)**

### **Equations:**
- \( L^R_Q(\theta) = \mathbb{E}[(Q^R_{\text{target}} - Q^R_\theta)^2] \)
- \( L^R_V(\psi) = \mathbb{E}[V^R_{\text{target}} - V^R_\psi]^2 \)

### **Code Implementation:**
```python
def reward_loss_piecewise_fn(self, critic_params, batch):
    target_q = self.reward_piecewise_target(batch)
    qs = self.critic.apply_fn({"params": critic_params}, batch["observations"], batch["actions"])
    loss = ((qs - target_q) ** 2).mean()
    return loss, {"reward_loss": loss, "q": qs.mean()}
```
- **Location:** Method `reward_loss_piecewise_fn` in `CBF` class.
- **Implements:** Reward critic loss.

```python
def value_loss_piecewise_fn(self, value_params, batch):
    qh = self.safe_critic.apply_fn({"params": self.safe_critic.params}, batch["observations"], batch["actions"]).max(axis=0)
    qs = self.critic.apply_fn({"params": self.critic.params}, batch["observations"], batch["actions"]).min(axis=0)
    mask_unsafe = (qh > 0)
    mask_safe = (qh <= 0)
    target_v = mask_safe * qs + mask_unsafe * (self.r_min / (1 - self.discount) - qh)
    v = self.value.apply_fn({"params": value_params}, batch["observations"])
    loss = ((v - target_v) ** 2).mean()
    return loss, {"value_loss": loss, "v": v.mean()}
```
- **Location:** Method `value_loss_piecewise_fn` in `CBF` class.
- **Implements:** Value loss for reward critic.

---

## 5. **Penalty Term for Unsafe Actions (Eq. 7, 8 from formulation)**

### **Equations:**
- Penalty term added to loss for unsafe actions (indicator function).

### **Code Implementation:**
**Not directly implemented in the provided code.**  
- The code uses a piecewise target, but does **not** add a separate penalty term as in Eq. 7, 8.  
- If you want to implement the penalty, you would add an extra term to the loss function for unsafe actions.

---

## 6. **Expectile Regression for Value Update**

### **Equation:**
- Expectile loss for value update.

### **Code Implementation:**
```python
def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
```
- **Location:** Top of cbf.py
- **Implements:** Expectile regression for value update.

---

## 7. **Soft Updates (Target Networks)**

### **Equation:**
- Target networks updated via soft updates.

### **Code Implementation:**
```python
target_critic_params = optax.incremental_update(
    critic.params, self.target_critic.params, self.tau
)
target_critic = self.target_critic.replace(params=target_critic_params)
```
- **Location:** In update methods like `update_reward_critic`, `update_cbf`, etc.
- **Implements:** Soft update for target networks.
!