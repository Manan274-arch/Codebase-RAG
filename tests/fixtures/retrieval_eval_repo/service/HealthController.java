import org.springframework.web.bind.annotation.GetMapping;

class HealthController {
    @GetMapping("/health")
    public String readinessStatus() {
        return "ready";
    }
}
