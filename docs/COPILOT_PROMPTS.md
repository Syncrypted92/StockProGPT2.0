# Copilot Prompts

Copy and paste these prompts into GitHub Copilot Chat:

## 1. Repository Overview Analysis

```
@workspace Analyze this repository structure and create a comprehensive overview document. Include:
1. Technology stack and frameworks used
2. Main components and their purposes
3. Entry points (APIs, main applications, services)
4. Database components (tables, stored procedures, views)
5. External dependencies and integrations
6. Configuration and deployment aspects

Generate this as a markdown document with Mermaid diagrams showing the high-level architecture. Save the output as REPOSITORY_OVERVIEW.md in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 2. Business Process Discovery

```
@workspace Examine this codebase and identify all business processes. For each process found:
1. Determine the trigger (user action, scheduled job, event)
2. Map the complete flow from start to finish
3. Identify key business rules applied
4. Document data inputs and outputs
5. Show error handling and exception paths

Create a BUSINESS_PROCESSES.md document with Mermaid flowcharts for each process. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 3. Database Schema Documentation

```
@workspace Analyze all database-related code and create comprehensive database documentation including:
1. All tables with their columns, types, and constraints
2. Stored procedures with their parameters and purpose
3. Views and functions
4. Foreign key relationships
5. Indexes and their purposes
6. Triggers and their business logic

Generate DATABASE_SCHEMA.md with ER diagrams using Mermaid syntax. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 4. API Documentation Generation

```
@workspace Scan for all API endpoints and external service calls. Document:
1. All REST/SOAP/GraphQL endpoints with request/response examples
2. Authentication and authorization requirements
3. Input validation and business rules
4. Error responses and status codes
5. External APIs this system calls
6. API versioning and backward compatibility

Create API_DOCUMENTATION.md with sequence diagrams showing API interactions. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 5. Data Flow and Lineage Analysis

```
@workspace Trace data flow through this system and create data lineage documentation:
1. Identify all data entry points (APIs, file imports, databases)
2. Map data transformations and business rule applications
3. Track data storage and persistence
4. Document data consumption and exports
5. Show data archival and cleanup processes

Generate DATA_LINEAGE.md with detailed flow diagrams for each major data entity. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 6. Architecture Layer Identification

```
@workspace Analyze this codebase and organize components into architectural layers:
1. Presentation Layer: UI, controllers, API endpoints
2. Business Logic Layer: Services, business rules, workflows
3. Data Access Layer: Repositories, database operations
4. Infrastructure Layer: Utilities, external integrations
5. Cross-cutting Concerns: Logging, security, caching

Create SYSTEM_ARCHITECTURE.md showing how these layers interact. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 7. Security and Error Handling Analysis

```
@workspace Examine security implementation and error handling patterns:
1. Authentication and authorization mechanisms
2. Input validation and sanitization
3. Error handling patterns and logging
4. Security vulnerabilities and mitigations
5. Data protection and privacy measures

Generate SECURITY_AND_ERROR_HANDLING.md with security flow diagrams. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 8. Performance and Scalability Assessment

```
@workspace Analyze performance and scalability aspects:
1. Database query optimization and indexing
2. Caching strategies and implementation
3. Potential bottlenecks and performance issues
4. Scalability considerations and limitations
5. Resource usage patterns

Create PERFORMANCE_ANALYSIS.md with recommendations for improvements. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 9. Integration and Dependencies Mapping

```
@workspace Map all external integrations and dependencies:
1. Third-party APIs and services
2. Database connections and external data sources
3. File system dependencies
4. Configuration dependencies
5. Inter-service communication patterns

Generate INTEGRATIONS_AND_DEPENDENCIES.md with integration flow diagrams. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```

---

## 10. Deployment and Operations Documentation

```
@workspace Analyze deployment and operational aspects:
1. Configuration management and environment settings
2. Database migration and deployment scripts
3. Monitoring and logging implementation
4. Backup and recovery procedures
5. Maintenance and operational tasks

Create DEPLOYMENT_AND_OPERATIONS.md with deployment flow diagrams. Save the output in the generated-docs/[REPOSITORY-NAME]/ folder.
```
